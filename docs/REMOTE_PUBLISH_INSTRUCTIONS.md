# Apply, commit, tag, and publish this release

These commands must be run by a user authenticated to the GitHub repository.

```bash
git clone https://github.com/luvsachin/XAI-IDS-MultiDataset-Evaluation.git
cd XAI-IDS-MultiDataset-Evaluation

# Copy the contents of this release-candidate folder into the repository root,
# preserving 03_Code/, results_summary/, 04_Results/, and docs/ paths.

python 03_Code/scripts/51_validate_frozen_release.py

git checkout -b release/raise-ids-v1.0.0
git add 03_Code/scripts/50_freeze_authoritative_release.py \
        03_Code/scripts/51_validate_frozen_release.py \
        results_summary 04_Results/frozen_release docs README_RAISE_IDS_RELEASE.md requirements_release.txt
git commit -m "Freeze RAISE-IDS v1.0.0 evidence lineage"
git tag -a raise-ids-v1.0.0 -m "RAISE-IDS frozen submission release"
git push origin release/raise-ids-v1.0.0
git push origin raise-ids-v1.0.0
```

After review, merge the branch to `main` and create a GitHub Release from tag `raise-ids-v1.0.0`. For an archival DOI, connect the GitHub repository to Zenodo and archive the tagged release.
