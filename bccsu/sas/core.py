import saspy
from pathlib import Path
import pandas as pd
import subprocess
import re

class Session(saspy.SASsession):
    @staticmethod
    def lazy_loader(func):
        """Decorator to ensure lazy loading is handled."""

        def wrapper(self, *args, **kwargs):
            if not self._lazy_loaded:
                self._initialize()
            return func(self, *args, **kwargs)

        return wrapper

    def __init__(self, *args, **kwargs):
        self._lazy_loaded = False

    def _initialize(self, *args, **kwargs):
        sas_config_path = (Path(__file__).parent / './sas_config.py').resolve()
        try:
            from bccsu.sas.load_session import pre, post
            error = None
            try:
                pre()
                _session = super().__init__(cfgfile=sas_config_path)
            except Exception as e:
                error = e
            finally:
                post()
                if error:
                    raise error

        except ImportError:
            _session = super().__init__(cfgfile=sas_config_path)

        self._lazy_loaded = True
        with open(Path(__file__).parent / 'sas_macros.sas', 'r') as f:
            macro_code = f.read()
        self.HTML_Style = 'listing'
        self.submit(rf'{macro_code}'
                    rf'ods exclude all;')
        self.HTML_Style = 'listing'
        return session

    @lazy_loader
    def submit(self, *args, **kwargs):
        return super().submit(*args, **kwargs)

    @lazy_loader
    def df2sd(self, *args, **kwargs):
        return super().df2sd(*args, **kwargs)


session = Session()


def proc_delete(tables):
    session.submit(rf'''
    proc delete data={' '.join(tables)};
    run;
    ''')


def base_command_pythonize(function):
    temp_data_name = 'TEMP_DATA'

    def wrapper(*args, **kwargs):
        df = args[0]

        wrapper_vars = {'v': False,
                        'outputs': [],
                        'save_as': '',
                        'show_sas_code': True,
                        'sas_results': '',
                        'prefix': '',
                        'code_only': False,
                        'name': ''}

        for i in wrapper_vars.keys():
            try:
                wrapper_vars[i] = kwargs[i]
            except KeyError:
                pass

        if wrapper_vars['code_only']:
            return function(*args, **kwargs) + f"%add_tables({output_serializer(kwargs.get('outputs'), prefix=kwargs.get('prefix'), out_tables_only=True)});"

        results = {}
        if not wrapper_vars['sas_results']:
            session.df2sd(df, temp_data_name, 'WORK')

            if len(args) > 1:
                args = args[1:]
            else:
                args = []
            s = function(*args, **kwargs)

            if wrapper_vars['show_sas_code']:
                print(s)

            save_as = wrapper_vars['save_as']
            if save_as:
                s = rf'''
                ods excel file="{Path(save_as).resolve()}";
                {s}
                ods excel close;
                '''

            r = session.submit(s)

            if wrapper_vars['v']:
                print(r['LOG'])
            if wrapper_vars['name']:
                name = wrapper_vars['name']
                dir = Path('./html')
                dir.mkdir(exist_ok=True)
                already_have = []

                for f in dir.glob(f"{name}*.html"):
                    search = re.search(rf"{name}_([\d])*?\.html", f.name)
                    if search:
                        already_have.append(int(search.group(1)))
                current_version = max(already_have) + 1 if already_have else 1
                html = r['LST']
                # replace the title
                html = re.sub(r'<title>.*?</title>', f'<title>{name}_{current_version}</title>', html)
                output = dir / f"{name}_{current_version}.html"
                with open(output, 'w') as f:
                    f.write(html)
                subprocess.run(['start', str(output)], shell=True)


            for i in wrapper_vars['outputs']:
                table_list = [t[0] for t in session.list_tables()]
                prefix = False
                for t in table_list:
                    if t[:1] == '_':
                        prefix = True

                if prefix:
                    results[i.lower()] = session.sasdata(f'_{i}', 'WORK').to_df()
                    proc_delete([f'_{i}'])
                else:
                    results[i.lower()] = session.sasdata(f'{i}', 'WORK').to_df()
                    proc_delete([i])

                # saspy list available tables in session
                # session.list_tables()
            # proc_delete(wrapper_vars['outputs'])
        else:
            path = wrapper_vars['sas_results']
            prefix = wrapper_vars['prefix']
            prefixed_outputs = [f'{prefix}_{i}' for i in wrapper_vars['outputs']]
            results = pd.read_excel(path, sheet_name=None)
            results = {key.lower()[len(prefix) + 1:]: value for key, value in results.items() if
                       key.lower() in prefixed_outputs}
        return results

    return wrapper


def output_serializer(outputs, prefix='', out_tables_only=False):
    if not outputs:
        return ''
    s = 'ods output' if not out_tables_only else ''
    entries = []
    for i in outputs:
        i_out = f'{prefix}_{i}' if prefix else i
        if out_tables_only:
            entries.append(i_out)
        else:
            entries.append(f'{i}={i_out}')
    if out_tables_only:
        s = ' '.join(entries)
    else:
        s += f' {" ".join(entries)};'
    return s
