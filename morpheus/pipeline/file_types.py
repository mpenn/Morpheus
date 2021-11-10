import enum
import os


@enum.unique
class FileTypes(str, enum.Enum):
    """The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use 'auto' to determine
    from the file extension."""
    Auto = "auto"
    Json = "json"
    Csv = "csv"


def determine_file_type(filename: str) -> FileTypes:
    # Determine from the file extension
    ext = os.path.splitext(filename)

    # Get the extension without the dot
    ext = ext[1].lower()[1:]

    # Check against supported options
    if (ext == "json" or ext == "jsonlines"):
        return FileTypes.Json
    elif (ext == "csv"):
        return FileTypes.Csv
    else:
        raise RuntimeError("Unsupported extension '{}' with 'auto' type. 'auto' only works with: csv, json".format(ext))
