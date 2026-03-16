SYSTEM_PROMPT = (
    "You solve toy shell tasks. "
    "Return exactly one shell command. "
    "Do not include markdown, comments, or explanation. "
    "Assume the current working directory is the workspace root."
)

USER_PROMPT_TEMPLATE = """\
Task:
{task}

Filesystem summary:
{filesystem_summary}"""
