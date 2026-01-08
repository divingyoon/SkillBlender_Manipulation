# Copyright 2025 Enactic, Inc.
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

"""Register SkillBlender policy classes with rsl_rl."""

from __future__ import annotations


def register_rsl_rl():
    """Expose custom policies to rsl_rl via module injection."""
    from sbm.rl.actor_critic_hierarchical import ActorCriticHierarchical

    import rsl_rl.modules as rsl_modules

    rsl_modules.ActorCriticHierarchical = ActorCriticHierarchical
    if hasattr(rsl_modules, "__all__") and "ActorCriticHierarchical" not in rsl_modules.__all__:
        rsl_modules.__all__.append("ActorCriticHierarchical")

    try:
        import rsl_rl.runners.on_policy_runner as on_policy_runner

        on_policy_runner.ActorCriticHierarchical = ActorCriticHierarchical
    except Exception:
        pass

    try:
        import rsl_rl.runners.distillation_runner as distillation_runner

        distillation_runner.ActorCriticHierarchical = ActorCriticHierarchical
    except Exception:
        pass
