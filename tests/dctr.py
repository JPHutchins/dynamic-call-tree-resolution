# Copyright (c) 2026 JP Hutchins
# SPDX-License-Identifier: MIT

"""The installed ``dctr`` executable, run as a subprocess."""

import subprocess
import sys
from pathlib import Path


def dctr(*arguments: str) -> str:
	return subprocess.run(
		[str(Path(sys.executable).with_name("dctr")), *arguments],
		check=True,
		capture_output=True,
		text=True,
	).stdout
