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

Write-Host "Stopping existing remote backend if it is running"
$stopBackendCommand = "pkill -f 'uvicorn.*ui.backend.main:app' || true; fuser -k 8000/tcp 2>/dev/null || true; sleep 1"
ssh $sshHost "bash" "-lc" $stopBackendCommand

Write-Host "Updating remote repository on ${sshHost}"
$updateRepoCommand = "cd $remoteProjectRoot && git fetch --all --prune && git pull --ff-only"
ssh $sshHost "bash" "-lc" $updateRepoCommand

Write-Host "Syncing backend config to ${sshHost}:${remoteBackendConfig}"
scp "$localBackendConfig" "${sshHost}:${remoteBackendConfig}"

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
