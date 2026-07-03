# Caribbean and Surinamese culture models

BeatMaster includes transparent culture profiles for prompt conditioning and a registry for deployment-specific fine-tuned models. The repository does **not** falsely label generic weights as culturally authentic.

## Model registry

Set `CULTURE_MODEL_MAP` to a JSON object mapping a profile ID to a compatible generation model:

```env
CULTURE_MODEL_MAP={"suriname-kaseko":"your-org/kaseko-model","caribbean-soca":"your-org/soca-model"}
```

When a profile has a mapped model, generated asset metadata records:

- the exact model ID;
- `model_source: culture-model-map`;
- `fine_tuned_for_profile: true`;
- the culture profile and enhanced prompt.

Without a mapped model, BeatMaster uses the configured base model and records that the result used profile-based prompt conditioning only.

## Requirements before calling a model culturally authentic

1. **Rights and consent:** every recording must have documented permission for machine-learning use, performer consent, territory, duration and commercial scope.
2. **Community participation:** musicians, producers, language specialists and cultural organizations from the represented community must help define the dataset, tags and evaluation criteria.
3. **Provenance:** retain source ID, owner, license, performer consent, region, language, subgenre, instruments, recording year and withdrawal status.
4. **Balanced data:** avoid over-representing one decade, studio, artist, language or commercial style.
5. **Holdout evaluation:** keep artist-disjoint and recording-disjoint validation sets.
6. **Human review:** use qualified reviewers from the represented tradition to score rhythm, instrumentation, language, arrangement and stereotype risk.
7. **Memorization checks:** compare generated audio against training recordings with fingerprinting and nearest-neighbor analysis.
8. **Model card:** publish limitations, intended use, excluded uses, licensing and known failure modes.

## Recommended profile IDs

- `suriname-kaseko`
- `suriname-kawina`
- `suriname-baithak-gana`
- `caribbean-soca`
- `caribbean-calypso`
- `caribbean-reggae`
- `caribbean-dancehall`
- `caribbean-zouk-kompa`
- `caribbean-chutney`

## Dataset manifest

Use `training/dataset_manifest.schema.json` and validate manifests before training:

```bash
python training/validate_manifest.py training/my_dataset.json
```

The validator rejects missing consent, missing licenses, unknown profiles and withdrawn recordings.

## Fine-tuning

The runtime accepts any model compatible with the configured Transformers `MusicgenForConditionalGeneration` interface. Training code and weights are intentionally not fabricated: they depend on legally acquired data, compute, model licensing and community review. Register only models that genuinely meet the requirements above.
