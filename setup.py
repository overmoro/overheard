"""py2app build configuration for Overheard.app.

Note: pyproject.toml must NOT be present in the working directory when this
runs: setuptools auto-reads it and sets install_requires, which py2app 0.28+
forbids. build_app.sh handles this by temporarily renaming pyproject.toml.
"""

from setuptools import setup

APP = ["src/overheard/app.py"]

DATA_FILES = []

OPTIONS = {
    "semi_standalone": True,
    "argv_emulation": False,
    "iconfile": "icon/overheard.icns",
    "plist": {
        "CFBundleName": "Overheard",
        "CFBundleDisplayName": "Overheard",
        "CFBundleIdentifier": "au.com.overmoro.overheard",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        # Menu bar app, so no dock icon
        "LSUIElement": True,
        # Microphone permission
        "NSMicrophoneUsageDescription": (
            "Overheard needs microphone access to record meetings."
        ),
        # System audio capture via Core Audio process taps (macOS 14.4+).
        # Without this key the tap fails rather than prompting.
        "NSAudioCaptureUsageDescription": (
            "Overheard records the audio from your meeting so it can be "
            "transcribed on this Mac."
        ),
        # Notification style
        "NSUserNotificationAlertStyle": "alert",
        # Launched from Finder there is no LANG, so Python falls back to ASCII
        # and any dependency opening a UTF-8 file without saying so fails.
        # parakeet_mlx reads the model's config.json with a bare open(), so the
        # model failed to load with a UnicodeDecodeError that its own fallback
        # then reported as "model not found".
        "LSEnvironment": {
            "PYTHONUTF8": "1",
            "LC_CTYPE": "UTF-8",
        },
    },
    "packages": [
        "overheard",
        "rumps",
        "objc",
        "AppKit",
        "Foundation",
        "UserNotifications",
    ],
    "excludes": [
        # Large ML packages live in system site-packages; not bundled.
        # They are resolved at runtime via semi_standalone mode.
        #
        # This list is an instruction to py2app's modulegraph, not a record of
        # what the app imports. modulegraph follows transitive and
        # conditionally-guarded imports and does not understand a runtime
        # `if is_available()`, so a name belongs here whenever a dependency
        # edge could reach it, whether or not first-party code imports it.
        # torch and friends stayed installed in this environment after the
        # single-engine cut removed every first-party import of them, so
        # dropping them from this list would let any surviving edge pull them
        # into the bundle. Remove them only after a clean build_app.sh run
        # shows the bundle contents unchanged.
        "torch",
        "torchaudio",
        "whisperx",
        "pyannote",
        "transformers",
        "huggingface_hub",
    ],
}

setup(
    name="Overheard",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
