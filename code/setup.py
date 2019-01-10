from distutils.core import setup

setup(
    name='wflopg',
    version='',
    description="wflopg is code to do wind farm layout optimization using pseudo-gradients",
    long_description="""
    wflopg is code to do wind farm layout optimization using pseudo-gradients.
    """,
    author="Erik Quaeghebeur",
    author_email="E.R.G.Quaeghebeur@tudelft.nl",
    url='https://bitbucket.org/equaeghe/pseudo_gradients-code',
    packages=[
      'wflopg',
      'wflopg.create_turbine',
      'wflopg.create_wind'
    ]
)
