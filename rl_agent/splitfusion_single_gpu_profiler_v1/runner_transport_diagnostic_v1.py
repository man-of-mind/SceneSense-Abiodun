"""Retained runner with durable copies of owned modem logs before deletion."""
import json
import os
from pathlib import Path
import shutil


def main():
    from . import runner_cpu_partition_v1 as R
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as B
    output = Path(os.environ['SCENESENSE_PARTITION_OUTPUT'])
    remove = B._remove_owned_radio_scratch

    def preserve(radio_state, radio_namespace):
        state = Path(radio_state)
        try:
            authority = json.loads((output / 'RADIO_AUTHORITY.json').read_text())
            if state.resolve() != Path(authority['radio_state']).resolve():
                raise RuntimeError('Foreign radio state')
            saved = output / 'radio_logs'
            saved.mkdir(exist_ok=False)
            for name in ('gnb.log', 'ue.log', 'radio_materialization.json', 'ATTACHED_RADIO_STATE.json'):
                if (state / name).is_file():
                    shutil.copyfile(state / name, saved / name)
            (saved / 'SCOPE.json').write_text(json.dumps(dict(source=str(state),
                exact_bsr_crc_available=False, logging_changed=False)) + '\n')
        except Exception as exc:
            (output / 'RADIO_LOG_PRESERVATION_ERROR.json').write_text(json.dumps(dict(error=repr(exc))) + '\n')
        finally:
            result = remove(radio_state, radio_namespace)
        return result

    B._remove_owned_radio_scratch = preserve
    return R.main()


if __name__ == '__main__':
    raise SystemExit(main())
