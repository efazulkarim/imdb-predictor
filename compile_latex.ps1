# Automatic LaTeX Compilation Script for IMDb Predictor Report
# Save this as: compile_latex.ps1

Write-Host "🎬 IMDb Rating Predictor - LaTeX Auto-Compiler" -ForegroundColor Cyan
Write-Host ""

# Check if pdflatex is installed
Write-Host "Checking for LaTeX compiler..." -ForegroundColor Yellow
$latexFound = $false
try {
    $pdflatexPath = Get-Command pdflatex -ErrorAction Stop | Select-Object -ExpandProperty Source
    if ($pdflatexPath.Source) {
        $latexFound = $true
        Write-Host "✓ pdflatex found at: $($pdflatexPath.Source)" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ pdflatex not found" -ForegroundColor Red
}

# If not found, offer to install MiKTeX
if (-not $latexFound) {
    Write-Host ""
    Write-Host "pdflatex not installed. Would you like to install MiKTeX?" -ForegroundColor Yellow
    $response = Read-Host "Install MiKTeX now? [Y/N]: "
    
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host "Installing MiKTeX..." -ForegroundColor Yellow
        Write-Host "This may take several minutes. Please be patient." -ForegroundColor Cyan
        
        # Download MiKTeX installer
        $installerUrl = "https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/miktex-portable.exe"
        $installerPath = "$env:TEMP\miktex-installer.exe"
        
        try {
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
            Write-Host "Downloaded installer to: $installerPath" -ForegroundColor Green
            Write-Host "Running installer..." -ForegroundColor Yellow
            
            # Run installer silently
            Start-Process $installerPath -ArgumentList "--auto-install=yes" -Wait
            Write-Host "MiKTeX installed successfully!" -ForegroundColor Green
            
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";C:\Program Files\MiKTeX 2.9\miktex\bin\x64"
            [System.Environment]::SetEnvironmentVariable("Path", $env:Path, "Machine")
            
        } catch {
            Write-Host "Installation failed or was cancelled. Please install MiKTeX manually from: https://miktex.org/download" -ForegroundColor Red
            Write-Host ""
            Write-Host "After manual installation, run this script again." -ForegroundColor Yellow
            Read-Host "Press Enter to exit..."
            exit
        }
    } else {
        Write-Host "Installation cancelled. Exiting..." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        exit
    }
}

# Compile LaTeX document
if ($latexFound) {
    Write-Host ""
    Write-Host "Compiling LaTeX document..." -ForegroundColor Yellow
    
    $projectPath = "c:\Users\salma\Desktop\Research Methodology\imdb-predictor\SEU Research Final Template"
    $mainTex = "$projectPath\main.tex"
    $pdfOutput = "$projectPath\main.pdf"
    
    if (-not (Test-Path $mainTex)) {
        Write-Host "Error: main.tex not found at: $mainTex" -ForegroundColor Red
        Write-Host "Please check the path and try again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        exit
    }
    
    try {
        # Change to project directory
        Set-Location -Path $projectPath -ErrorAction Stop
        
        # Run pdflatex twice (for references)
        Write-Host "First compilation pass..." -ForegroundColor Cyan
        $compileResult = & pdflatex -interaction=nonstopmode main.tex 2>&1 | Select-String -Pattern "Output written" -Context 0, 5
        Write-Host $compileResult -ForegroundColor Green
        
        Write-Host "Second compilation pass..." -ForegroundColor Cyan
        $compileResult = & pdflatex -interaction=nonstopmode main.tex 2>&1 | Select-String -Pattern "Output written" -Context 0, 5
        Write-Host $compileResult -ForegroundColor Green
        
        # Check if PDF was created
        if (Test-Path $pdfOutput) {
            Write-Host ""
            Write-Host "✓ SUCCESS! PDF created at: $pdfOutput" -ForegroundColor Green
            Write-Host ""
            Write-Host "Opening PDF..." -ForegroundColor Cyan
            Start-Process $pdfOutput
            
            Write-Host ""
            Write-Host "PDF location: $pdfOutput" -ForegroundColor Yellow
        } else {
            Write-Host "✗ Error: PDF file not found after compilation." -ForegroundColor Red
            Write-Host "Please check the LaTeX log for errors." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "✗ Compilation failed with error:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host ""
        Write-Host "Please check for LaTeX syntax errors in main.tex" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Press Enter to exit..." -ForegroundColor Gray
$null = Read-Host
