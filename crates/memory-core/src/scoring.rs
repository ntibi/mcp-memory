use serde::Deserialize;

const Z: f64 = 1.96; // 95% confidence interval

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    pub relevance_weight: f64,
    pub confidence_weight: f64,
    pub recency_weight: f64,
    pub recency_half_life_days: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            relevance_weight: 0.6,
            confidence_weight: 0.25,
            recency_weight: 0.15,
            recency_half_life_days: 30.0,
        }
    }
}

pub struct Scorer {
    config: ScoringConfig,
}

impl Scorer {
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    pub fn score(&self, relevance: f64, helpful: u64, harmful: u64, age_days: f64) -> f64 {
        let confidence = wilson_score(helpful, harmful);
        let recency = (-2.0_f64.ln() / self.config.recency_half_life_days * age_days).exp();

        self.config.relevance_weight * relevance
            + self.config.confidence_weight * confidence
            + self.config.recency_weight * recency
    }
}

fn wilson_score(helpful: u64, harmful: u64) -> f64 {
    let n = (helpful + harmful) as f64;
    if n == 0.0 {
        return 0.5;
    }
    let p = helpful as f64 / n;
    let z2 = Z * Z;
    let numerator = p + z2 / (2.0 * n) - Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt());
    let denominator = 1.0 + z2 / n;
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn should_return_baseline_when_no_votes() {
        let scorer = Scorer::new(ScoringConfig::default());
        let score = scorer.score(1.0, 0, 0, 0.0);
        // wilson_score(0,0) = 0.5, exp(0) = 1.0
        let expected = 0.6 * 1.0 + 0.25 * 0.5 + 0.15 * 1.0;
        assert!((score - expected).abs() < EPSILON, "expected {expected}, got {score}");
    }

    #[test]
    fn should_rank_helpful_higher_than_harmful() {
        let scorer = Scorer::new(ScoringConfig::default());
        let helpful_score = scorer.score(0.8, 10, 0, 30.0);
        let harmful_score = scorer.score(0.8, 0, 10, 30.0);
        assert!(helpful_score > harmful_score, "helpful {helpful_score} should beat harmful {harmful_score}");
    }

    #[test]
    fn should_rank_recent_higher_than_old() {
        let scorer = Scorer::new(ScoringConfig::default());
        let recent = scorer.score(0.8, 5, 1, 0.0);
        let old = scorer.score(0.8, 5, 1, 365.0);
        assert!(recent > old, "recent {recent} should beat old {old}");
    }

    #[test]
    fn should_use_custom_weights_when_configured() {
        let config = ScoringConfig {
            relevance_weight: 1.0,
            confidence_weight: 0.0,
            recency_weight: 0.0,
            recency_half_life_days: 30.0,
        };
        let scorer = Scorer::new(config);
        let score = scorer.score(0.75, 100, 0, 0.0);
        assert!((score - 0.75).abs() < EPSILON, "expected 0.75, got {score}");
    }
}
