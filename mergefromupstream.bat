git remote add upstream https://github.com/virattt/ai-hedge-fund
git fetch upstream
git checkout main
git merge upstream/main
git commit -m "Merged changes from upstream"
git push origin main
