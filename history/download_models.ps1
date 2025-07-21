# PowerShell script to download models from hf-mirror.com
param(
    [string]$ModelName = "Qwen/Qwen2.5-0.5B-Instruct",
    [string]$OutputDir = ".\QWen\Qwen2.5-0.5B-Instruct"
)

# Create the output directory if it doesn't exist
if (-not (Test-Path -Path $OutputDir)) {
    Write-Host "Creating directory: $OutputDir"
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
}

# Construct the URL for hf-mirror.com
$repoPath = $ModelName -replace "/", "--"
$baseUrl = "https://hf-mirror.com/models--$repoPath/snapshots/"

# First, let's get the latest snapshot ID
Write-Host "Getting latest snapshot ID for $ModelName..."
try {
    $response = Invoke-WebRequest -Uri "https://hf-mirror.com/$ModelName/tree/main" -UseBasicParsing
    $snapshotPattern = "/models--$($repoPath -replace "/", "--")/snapshots/([a-zA-Z0-9]+)"
    $snapshotId = [regex]::Match($response.Content, $snapshotPattern).Groups[1].Value
    
    if (-not $snapshotId) {
        Write-Host "Failed to find snapshot ID. Using 'main' instead."
        $snapshotId = "main"
    } else {
        Write-Host "Found snapshot ID: $snapshotId"
    }
} catch {
    Write-Host "Error getting snapshot ID: $_"
    Write-Host "Using 'main' instead."
    $snapshotId = "main"
}

# List of common model files to download
$filesToDownload = @(
    "config.json",
    "configuration_qwen.py",
    "generation_config.json",
    "modeling_qwen.py",
    "qwen_model.safetensors",
    "qwen_model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_qwen.py"
)

# Download each file
foreach ($file in $filesToDownload) {
    $fileUrl = "https://hf-mirror.com/$ModelName/resolve/$snapshotId/$file"
    $outputPath = Join-Path -Path $OutputDir -ChildPath $file
    
    Write-Host "Downloading $file to $outputPath..."
    try {
        Invoke-WebRequest -Uri $fileUrl -OutFile $outputPath -UseBasicParsing
        Write-Host "Successfully downloaded $file"
    } catch {
        Write-Host "Error downloading $file: $_"
    }
}

Write-Host "Download process completed!"
