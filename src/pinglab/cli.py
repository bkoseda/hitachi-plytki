import argparse
from pathlib import Path
from .audio import Config


def main():
    parser=argparse.ArgumentParser(description='Ping Lab: lokalny warsztat analizy uderzeń')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Generuj oznaczone dane syntetyczne')
    demo.add_argument('--output',default='data/demo')
    run=sub.add_parser('analyze',help='Analiza manifestu i raport HTML')
    run.add_argument('manifest')
    run.add_argument('--output',required=True)
    run.add_argument('--annotations',help='JSON: ścieżka z manifestu -> lista sekund uderzeń')
    run.add_argument('--fmax',type=float,default=20000)
    run.add_argument('--tail',type=float,default=1.0)
    run.add_argument('--gap',type=float,default=.25)
    run.add_argument('--sensitivity',type=float,default=6.)
    run.add_argument('--evaluate',action='store_true')
    run.add_argument('--source-kind',choices=['client','synthetic','external'],default='client')
    run.add_argument('--family',choices=['physics','mfcc','mel'],default='physics')
    args=parser.parse_args()
    if args.command=='demo':
        from .demo import generate_demo
        print(generate_demo(args.output));return
    from .dataset import analyze_manifest
    from .learning import evaluate
    from .report import write_report
    out=Path(args.output)
    if (out/'run.json').exists():
        parser.error('Katalog zawiera analizę. Wybierz nowy --output, aby zachować wyniki.')
    config=Config(tail_s=args.tail,min_gap_s=args.gap,sensitivity=args.sensitivity)
    features,qc=analyze_manifest(args.manifest,out,config,args.fmax,args.annotations)
    # Synthetic demo cannot accidentally become a client-quality claim.
    source=args.source_kind
    if not features.empty and features.environment.eq('synthetic').any():
        source='synthetic'
    assessment=None
    if args.evaluate:
        try:
            assessment=evaluate(features,family=args.family)
        except ValueError as ex:
            print(f'Ocena pominięta: {ex}')
    print(write_report(features,qc,out/'report.html',assessment,source_kind=source))
    print(f'{len(features)} uderzeń. Sprawdź quality.csv i odsłuch przed interpretacją.')


if __name__=='__main__':
    main()
