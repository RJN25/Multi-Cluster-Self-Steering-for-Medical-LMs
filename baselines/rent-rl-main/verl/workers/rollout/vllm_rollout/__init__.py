# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

def get_version(pkg: str):
    try:
        detected_version = version(pkg)
    except PackageNotFoundError:
        return None

    if "ROCM_PATH" in os.environ:
        import re

        match = re.match(r"(\d+\.\d+\.?\d*)", detected_version)
        if match is not None:
            detected_version = match.group(1)

    return detected_version


package_name = "vllm"
package_version = get_version(package_name)

###
if package_version is None:
    raise RuntimeError(
        "vLLM is required but was not found. Please install the `vllm` package so "
        "that its version can be detected."
    )

try:
    parsed_version = Version(package_version)
except InvalidVersion as exc:
    raise RuntimeError(
        f"Unable to parse installed vLLM version '{package_version}'."
    ) from exc

if parsed_version <= Version("0.6.3"):
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  # noqa: F401
    from .vllm_rollout import vLLMRollout  # noqa: F401
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
