$ErrorActionPreference = "Stop"

$localPort = 8000
$remoteHost = "127.0.0.1"
$remotePort = 8000
$sshHost = "node2"
$remoteProjectRoot = "/home/zhy/opengauss/tools/new_serverless_predictor"
$remotePython = "/home/zhy/miniconda3/envs/zhy_env/bin/python"
$localBackendConfig = Resolve-Path "$PSScriptRoot/../backend/config.yaml"
$remoteBackendConfig = "$remoteProjectRoot/ui/backend/config.yaml"
$localGaussRunner = Resolve-Path "$PSScriptRoot/../backend/services/gauss_runner.py"
$remoteGaussRunner = "$remoteProjectRoot/ui/backend/services/gauss_runner.py"
$localSettings = Resolve-Path "$PSScriptRoot/../backend/settings.py"
$remoteSettings = "$remoteProjectRoot/ui/backend/settings.py"
$localDatasetLoader = Resolve-Path "$PSScriptRoot/../backend/services/dataset_loader.py"
$remoteDatasetLoader = "$remoteProjectRoot/ui/backend/services/dataset_loader.py"
$backend = $null
$tunnel = $null

function Test-BackendHealth {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$localPort/api/health" -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-RemoteBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [int]$TimeoutSeconds = 0,
        [switch]$ContinueOnError
    )

    $scriptName = "predictor-ui-$([Guid]::NewGuid().ToString('N')).sh"
    $localScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) $scriptName
    $remoteScriptPath = "/tmp/$scriptName"
    $normalizedScript = ($Script -replace "`r`n", "`n" -replace "`r", "`n")
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    [System.IO.File]::WriteAllText($localScriptPath, $normalizedScript, $utf8NoBom)
    try {
        scp "$localScriptPath" "${sshHost}:${remoteScriptPath}"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to copy remote script with exit code $LASTEXITCODE"
        }

        if ($TimeoutSeconds -gt 0) {
            $sshProcess = Start-Process -FilePath "ssh" -ArgumentList @($sshHost, "bash", $remoteScriptPath) -NoNewWindow -PassThru
            $completed = $sshProcess.WaitForExit($TimeoutSeconds * 1000)
            if (-not $completed) {
                Stop-Process -Id $sshProcess.Id -Force
                $remoteExitCode = 124
            }
            else {
                $sshProcess.Refresh()
                $remoteExitCode = $sshProcess.ExitCode
                if ($null -eq $remoteExitCode) {
                    $remoteExitCode = 0
                }
            }
        }
        else {
            ssh $sshHost "bash" "$remoteScriptPath"
            $remoteExitCode = $LASTEXITCODE
        }

        ssh $sshHost "rm" "-f" "$remoteScriptPath" | Out-Null
        if ($remoteExitCode -ne 0) {
            if ($ContinueOnError) {
                Write-Warning "Remote command skipped/failed with exit code $remoteExitCode"
                return $false
            }
            throw "Remote command failed with exit code $remoteExitCode"
        }
        return $true
    }
    finally {
        if (Test-Path $localScriptPath) {
            Remove-Item $localScriptPath -Force
        }
    }
}

Write-Host "Stopping existing remote backend if it is running"
$stopBackendCommand = @'
set +e
echo "Stopping processes on port __REMOTE_PORT__"
command -v fuser >/dev/null 2>&1 && fuser -k __REMOTE_PORT__/tcp 2>/dev/null
command -v lsof >/dev/null 2>&1 && lsof -ti tcp:__REMOTE_PORT__ | xargs -r kill -9 2>/dev/null
pgrep -f "uvicorn.*ui[.]backend[.]main:app" | xargs -r kill -9 2>/dev/null
sleep 1
'@
$stopBackendCommand = $stopBackendCommand.Replace("__REMOTE_PORT__", [string]$remotePort)
Invoke-RemoteBash $stopBackendCommand

Write-Host "Updating remote repository on ${sshHost}"
$updateRepoCommand = @'
set -e
remote_project_root="__REMOTE_PROJECT_ROOT__"
if git -C "$remote_project_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Remote git root: $(git -C "$remote_project_root" rev-parse --show-toplevel)"
  echo "Remote git branch: $(git -C "$remote_project_root" branch --show-current)"
  if ! git -C "$remote_project_root" diff --quiet -- ui/backend/config.yaml ui/backend/settings.py ui/backend/services/dataset_loader.py ui/backend/services/gauss_runner.py; then
    echo "Stashing remote local backend edits before pull"
    git -C "$remote_project_root" stash push -m "predictor-ui-auto-stash-before-pull" -- ui/backend/config.yaml ui/backend/settings.py ui/backend/services/dataset_loader.py ui/backend/services/gauss_runner.py
  fi
  git -C "$remote_project_root" fetch --all --prune
  git -C "$remote_project_root" pull --ff-only
else
  echo "Skip git pull: $remote_project_root is not a git repository"
fi
'@
$updateRepoCommand = $updateRepoCommand.Replace("__REMOTE_PROJECT_ROOT__", $remoteProjectRoot)
Invoke-RemoteBash $updateRepoCommand -TimeoutSeconds 20 -ContinueOnError | Out-Null

Write-Host "Syncing backend config to ${sshHost}:${remoteBackendConfig}"
scp "$localBackendConfig" "${sshHost}:${remoteBackendConfig}"

Write-Host "Syncing backend settings to ${sshHost}:${remoteSettings}"
scp "$localSettings" "${sshHost}:${remoteSettings}"

Write-Host "Syncing dataset loader to ${sshHost}:${remoteDatasetLoader}"
scp "$localDatasetLoader" "${sshHost}:${remoteDatasetLoader}"

Write-Host "Syncing backend runner to ${sshHost}:${remoteGaussRunner}"
scp "$localGaussRunner" "${sshHost}:${remoteGaussRunner}"

Write-Host "Starting remote backend on ${sshHost}:${remotePort}"

$remoteBackendCommand = "cd $remoteProjectRoot && PYTHON=$remotePython ui/run_backend.sh"
$backendArgs = @(
    $sshHost,
    "bash",
    "-lc",
    "'$remoteBackendCommand'"
)

$backend = Start-Process -FilePath "ssh" -ArgumentList $backendArgs -NoNewWindow -PassThru

Write-Host "Starting SSH tunnel: localhost:$localPort -> ${sshHost}:${remoteHost}:$remotePort"

$sshArgs = @(
    "-N",
    "-L", "${localPort}:${remoteHost}:${remotePort}",
    $sshHost
)

$tunnel = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -NoNewWindow -PassThru

try {
    Start-Sleep -Seconds 2
    if ($tunnel.HasExited) {
        throw "SSH tunnel exited early with code $($tunnel.ExitCode). Check whether 'ssh node2' works."
    }

    Write-Host "Waiting for backend health check..."
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        if (Test-BackendHealth) {
            $ready = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $ready) {
        throw "Backend health check failed at http://localhost:$localPort/api/health. Check the remote backend logs."
    }

    Write-Host "Backend and SSH tunnel are ready. Starting Vite dev server..."
    npm run dev
}
finally {
    if ($null -ne $tunnel -and -not $tunnel.HasExited) {
        Write-Host "Stopping SSH tunnel..."
        Stop-Process -Id $tunnel.Id -Force
    }
    if ($null -ne $backend -and -not $backend.HasExited) {
        Write-Host "Stopping remote backend SSH session..."
        Stop-Process -Id $backend.Id -Force
    }
}
