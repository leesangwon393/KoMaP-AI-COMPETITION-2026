# Repository working rules

- Project code: src/. Keep shared modules flat; retain model1/model2/model3 packages where required for incompatible imports.
- Reusable shell launch workflows: script/. Do not create wrappers for a single command.
- Configurations, environment descriptions, metadata and data links: resources/.
- External repositories and public initialization weights: Thirdparty/. Do not modify external source in place; adapted code belongs in src/.
- All generated outputs and trained checkpoints: run/. Never commit data or large weights.
- Use existing environments; do not create an environment automatically.
- Do not modify or duplicate shared dataset originals. Use direct references or symlinks.
- Inspect existing implementation before adding files. Prefer minimal changes without unnecessary wrappers or abstraction.
- Final training/inference code lives in src/. Experimental history in docs/ is Markdown only; original experiment code and releases are preserved on archive/experiments-before-cleanup-20261015. Do not reintroduce historical runners into main.
