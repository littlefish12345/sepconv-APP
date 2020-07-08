# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['sepconv-APP.py'],
             binaries=[],
             datas=[],
             hiddenimports=['fastrlock','fastrlock.rlock','cupy.core.flags','cupy.core._routines_indexing','cupy.core._dtype','cupy.core._scalar','cupy.core._ufuncs','cupy.core._routines_sorting','pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='sepconv-APP',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='sepconv-APP')
