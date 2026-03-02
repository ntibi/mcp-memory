use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    pub relevance_weight: f64,
    pub confidence_weight: f64,
    pub recency_weight: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            relevance_weight: 0.6,
            confidence_weight: 0.25,
            recency_weight: 0.15,
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
        let confidence = if helpful + harmful == 0 {
            0.5
        } else {
            (helpful as f64) / ((helpful + harmful) as f64)
        };
        let recency = 1.0 / (1.0 + age_days);

        self.config.relevance_weight * relevance
            + self.config.confidence_weight * confidence
            + self.config.recency_weight * recency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn should_return_baseline_when_no_votes() {
        let scorer = Scorer::new(ScoringConfig::default());
        let score = scorer.score(1.0, 0, 0, 0.0);
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
        };
        let scorer = Scorer::new(config);
        let score = scorer.score(0.75, 100, 0, 0.0);
        assert!((score - 0.75).abs() < EPSILON, "expected 0.75, got {score}");
    }
}
