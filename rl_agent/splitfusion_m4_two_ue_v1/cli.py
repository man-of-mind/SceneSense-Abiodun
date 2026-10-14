"""Single M4 preparation, preflight, smoke, capture, analysis and recovery CLI."""
import argparse
import json
from pathlib import Path
import sys

from .identity import require


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('preflight'); p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--host',action='store_true',help='read-only live prerequisites, including unchanged disk gate')
    p=sub.add_parser('check'); p.add_argument('--output',type=Path,required=True)
    for name in ('smoke','capture'):
        p=sub.add_parser(name); p.add_argument('--preparation',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--execute-token',required=True,help='requires separate user authorization; never used by offline tests')
    p=sub.add_parser('supervised-child',help='internal: the CARLA client child of smoke/capture')
    p.add_argument('mode',choices=('smoke','capture')); p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--execute-token',required=True)
    p=sub.add_parser('analyze'); p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('cleanup'); p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--execute-owned',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            from .configuration import prepare
            result=prepare(args.output)
            print(json.dumps(dict(status='PREPARED_NOT_LIVE_QUALIFIED',sha256=result['sha256']))); return 0
        if args.command=='preflight':
            from .preflight import report
            print(json.dumps(report(args.preparation,host=args.host),sort_keys=True)); return 0
        if args.command=='check':
            from .composed_checks import run
            return run(args.output)
        if args.command in ('smoke','capture'):
            require(args.execute_token=='M4_TWO_UE_'+args.command.upper()+'_SEPARATELY_AUTHORIZED',
                    'wrong execute token; no live startup')
            import faulthandler
            faulthandler.enable(all_threads=True)
            # Parent never loads carla/torch; the client runs as its child.
            from .supervisor import run
            return run(args.preparation,args.output,smoke=args.command=='smoke',token=args.execute_token)
        if args.command=='supervised-child':
            require(args.execute_token=='M4_TWO_UE_'+args.mode.upper()+'_SEPARATELY_AUTHORIZED',
                    'wrong execute token; no live startup')
            import faulthandler
            faulthandler.enable(all_threads=True)  # stderr until host.run moves it into the capture
            from .host import run
            rc=run(args.preparation,args.output,smoke=args.mode=='smoke',carla_stop_delegated=True)
            from . import phases
            phases.current().event('cli.return',returncode=rc)
            return rc
        if args.command=='analyze':
            from .analysis import analyze
            result=analyze(args.preparation,args.capture,args.output)
            print(json.dumps(result,sort_keys=True)); return 0
        from .resources import recover
        print(json.dumps(recover(args.capture,execute=args.execute_owned),sort_keys=True)); return 0
    except Exception as error:
        print(json.dumps(dict(status='REFUSED_OR_FAILED',error=type(error).__name__+': '+str(error))),file=sys.stderr)
        return 2

if __name__=='__main__': raise SystemExit(main())
