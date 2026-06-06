__appname__ = "自进化侦测"
__appdescription__ = "Self-Evolving Detection · 自进化目标检测系统"
__version__ = "4.0.0-beta.5"
__url__ = "https://github.com/Luoxr520/code"

CLI_HELP_MSG = """
    Usage: ser [COMMAND] [OPTIONS]

    Available Commands:
        help              Show this help message
        checks            Display system and package information
        version           Show version information
        config            Show config file path
        convert           Run conversion tasks

    Launch Options:
        ser                                    Launch the GUI application
        ser --filename IMAGE                   Open specific image/folder
        ser --output DIR                       Set output directory
        ser --config FILE                      Use custom config file
        ser --reset-config                     Reset Qt config
        ser --qt-image-allocation-limit 1024   Set Qt image allocation limit to 1024 MB

    Conversion Tasks:
        ser convert                            List all conversion tasks
        ser convert --task <task>              Show help for a specific task
        ser convert --task <task> [options]    Run conversion

    Examples:
        1. Launch the app:
            ser

        2. Open an image:
            ser --filename /path/to/image.jpg

        3. Check system information:
            ser checks

        4. Show version:
            ser version

        5. List all conversion tasks:
            ser convert

        6. Show help for a conversion task:
            ser convert --task yolo2xlabel

        7. Convert YOLO to XLABEL:
            ser convert --task yolo2xlabel --mode detect --images ./images --labels ./labels --output ./output --classes classes.txt

    For more options, use: ser --help
"""


def __getattr__(name):
    if name == "__preferred_device__":
        from anylabeling.views.common.device_manager import (
            get_preferred_device,
        )

        return get_preferred_device()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
