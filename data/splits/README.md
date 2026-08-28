# Train / Validation / Test Splits

Following modern benchmark design and lessons from previous ESL literature (where signer-independent accuracy exhibited significant degradation), dataset splits in Tereguwami are strictly partitioned.

## Split Protocols

1. **Signer-Independent Split (Primary / Headline Benchmark)**
   - Train Split: ~70% of signers
   - Validation Split: ~15% of signers
   - Test Split: ~15% of signers
   - *Rule*: Zero signer overlap between train, val, and test splits.

2. **Signer-Dependent Split (Secondary / Reference Comparison)**
   - Maintained specifically for direct apples-to-apples comparison against historical isolated-sign baselines.
