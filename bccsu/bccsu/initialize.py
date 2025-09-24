# Generate a basic project structure

# With git initialize new repo with: git remote add origin git@github.com:BCCSU/CG_{project_name}.git
import jinja2
from pathlib import Path
import shutil

def copy_contents(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for entry in src.iterdir():  # includes hidden files
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)


# project_name = input('Enter project name: ')
project_name = 'CG_0044'

PROJECTS_ROOT = Path.home() / 'BCCSU'
PROJECT_FOLDER = PROJECTS_ROOT / project_name
TEMPLATE_ROOT = Path(__file__).parent / 'project_template'

if PROJECT_FOLDER.exists():
    raise FileExistsError(f'Project folder {PROJECT_FOLDER} already exists.')

PROJECT_FOLDER.mkdir()

copy_contents(TEMPLATE_ROOT, PROJECT_FOLDER)

def template_file(path, data=None, outname=None):
    with open(TEMPLATE_ROOT / path, 'r') as f:
        template = jinja2.Template(f.read())
    if data is None:
        data = {}
    doc = template.render(**data)

    output = PROJECT_FOLDER / path
    if outname is not None:
        (PROJECT_FOLDER / path).unlink()
        output = output.parent / outname


    with open(output, "w", encoding="utf-8") as f:
        f.write(doc)

template_file('.idea/template.iml',
              {"project_path": f'BCCSU/{PROJECT_FOLDER.name}'},
              outname=f'{project_name}.iml')
template_file('.idea/modules.xml', {'project_name': project_name})
template_file('.idea/workspace.xml', {'project_name': project_name})
template_file('.git/config', {'project_name': project_name})

# C:\Users\cgrant\PycharmProjects\bccsu-tools\bccsu\bccsu\project_template\.idea\CG_0045.iml
# rename and PycharmProjects/bccsu-tools ->

