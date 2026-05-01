$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$logOut = Join-Path $root "server.out.log"
$logErr = Join-Path $root "server.err.log"
$healthUrl = "http://127.0.0.1:8000/health"

function Test-PythonRuntime {
    param([string]$PythonPath)

    try {
        & $PythonPath -c "import fastapi, uvicorn, joblib, sklearn, sentence_transformers" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$candidates = @(
    (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"),
    "python"
)

$python = $null
foreach ($candidate in $candidates) {
    if (($candidate -eq "python" -or (Test-Path $candidate)) -and (Test-PythonRuntime $candidate)) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    Write-Host "No Python runtime with FastAPI/Uvicorn was found."
    Write-Host "Run: pip install -r backend\requirements.txt"
    exit 1
}

# Some Windows shells expose both Path and PATH, which can make Start-Process fail.
$processEnv = [Environment]::GetEnvironmentVariables("Process")
if ($processEnv.Contains("Path") -and $processEnv.Contains("PATH")) {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if (-not $pathValue) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
}

"" | Out-File -FilePath $logOut -Encoding utf8
"" | Out-File -FilePath $logErr -Encoding utf8

$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -PassThru

for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing $healthUrl -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "CV Analyzer AI is running at http://127.0.0.1:8000/app"
            Write-Host "Process ID: $($process.Id)"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "Server did not become ready in time."
Write-Host "Check logs:"
Write-Host "  $logOut"
Write-Host "  $logErr"
exit 1
