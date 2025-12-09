# PowerShell script to run the complete IMDb Rating Predictor pipeline
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Running IMDb Rating Predictor - Complete Pipeline" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

python imdb_rating_predictor.py

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")




