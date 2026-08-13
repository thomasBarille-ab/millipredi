# Configure le Planificateur de tâches Windows pour lancer auto_update.py chaque jour à 9h
# Lancer en tant qu'administrateur : .\setup_scheduler.ps1

$projectDir = "C:\Users\thoma\Desktop\euromillions-ml"
$python     = "$projectDir\venv\Scripts\python.exe"
$script     = "$projectDir\auto_update.py"
$taskName   = "EuroMillions-ML-AutoUpdate"

$action  = New-ScheduledTaskAction -Execute $python -Argument $script -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force

Write-Host "✅ Tâche '$taskName' créée — s'exécutera chaque jour à 09h00."
Write-Host "   Lundi/Jeudi   → prédictions générées"
Write-Host "   Mercredi/Sam  → résultats récupérés"
Write-Host ""
Write-Host "Pour supprimer : Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
