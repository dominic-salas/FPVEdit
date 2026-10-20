; Inno Setup 6 script: turns the standalone build (dist\FPVEdit, from
; scripts/build_windows.py) into one installer. The release workflow runs it as
;   iscc /DAppVersion=1.0.0 packaging\installer.iss
; Installs per user by default (no admin rights needed), or for all users. It also
; installs an uninstaller: Windows Settings > Apps > FPVEdit > Uninstall, or
; "Uninstall FPVEdit" in the Start menu.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
; Windows knows the app by this id (also from before the rename, as "FPV Editor"): an
; installer with the same id upgrades it. Tests build with another id (/DAppGuid=...), so
; they never touch a real installation.
#ifndef AppGuid
  #define AppGuid "{{6F0B5C2E-4D7A-4E7B-9A51-2C3E8F1D7A40}"
#endif

[Setup]
AppId={#AppGuid}
AppName=FPVEdit
AppVersion={#AppVersion}
AppPublisher=Dominic Salas
DefaultDirName={autopf}\FPVEdit
DefaultGroupName=FPVEdit
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=FPVEdit-{#AppVersion}-setup
SetupIconFile=..\build\windows\fpvedit.ico
UninstallDisplayIcon={app}\FPVEdit.exe
UninstallDisplayName=FPVEdit
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\FPVEdit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Left over from versions called "FPV Editor" (an upgrade installs into the same folder).
Type: files; Name: "{app}\FPV Editor.exe"
Type: files; Name: "{app}\fpv-editor.exe"
Type: files; Name: "{app}\licenses\FPV-Editor-LICENSE.txt"
Type: files; Name: "{group}\FPV Editor.lnk"
Type: files; Name: "{autodesktop}\FPV Editor.lnk"

[Icons]
Name: "{group}\FPVEdit"; Filename: "{app}\FPVEdit.exe"
Name: "{group}\Uninstall FPVEdit"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FPVEdit"; Filename: "{app}\FPVEdit.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FPVEdit.exe"; Description: "Start FPVEdit"; Flags: nowait postinstall skipifsilent

[Code]
// The app keeps the user's shot ratings, saved measurements (to find shots again
// quickly), picture scores and a log in the local app-data folder, and its settings in
// the registry. Uninstalling removes the program; these are offered for removal too
// (asked, never silently; No is the default). Videos and montages are never touched.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data, OldData: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{localappdata}\FPVEdit');
    OldData := ExpandConstant('{localappdata}\fpv-editor');  // before the rename
    if (DirExists(Data) or DirExists(OldData)) and (not UninstallSilent) then
      if MsgBox('Also remove your shot ratings, saved measurements, settings and log (' +
                Data + ')? Your videos and montages are not affected.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(Data, True, True, True);
        DelTree(OldData, True, True, True);
        RegDeleteKeyIncludingSubkeys(HKEY_CURRENT_USER, 'Software\FPVEdit');
      end;
  end;
end;
