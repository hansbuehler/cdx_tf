#@echo off
REM before using this script, update the version string in setup.py and __init__.py

echo =====================================================================================
echo PIP build
echo =====================================================================================

REM https://packaging.python.org/tutorials/packaging-projects/
cd C:\Users\hansb\iCloudDrive\Python3\packages\cdx_tf
if exist dist rmdir /Q /S dist
mkdir dist
python setup.py sdist bdist_wheel
python -m twine upload dist/*
rmdir /Q /S dist

echo =====================================================================================
echo Conda install: uninstall; build; install
echo =====================================================================================

REM https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html#troubleshooting
if exist conda rmdir /Q /S conda
mkdir conda
copy conda_exists.py conda
copy conda_modify_yaml.py conda
cd conda
python -m conda_exists.py
if not %ERRORLEVEL% == 0 goto NOTFOUND
	echo Removing existing conda package from installation. That may take a while
	call conda uninstall -y cdx_tf
	goto FOUND
:NOTFOUND
	echo No existing cdx_tf installation found
:FOUND
echo Generating new conda skeleton package
call conda skeleton pypi cdx_tf
echo Making package platform independent
python -m conda_modify_yaml.py cdx_tf/meta.yaml
echo Building package. That may take a while
call conda build cdx_tf
echo Cleaning up
call conda build purge
cd ..
rmdir /Q /S conda
echo Attempting conda install
call conda install -y cdx_tf -c hansbuehler

echo =====================================================================================
echo GIT upload
echo =====================================================================================

echo GIT upload
python git_message.py >.tmp.txt
set /p MESSAGE=< .tmp.txt
del /q .tmp.txt
REM echo Python test showed %MESSAGE%
git commit -a -m "%MESSAGE%"
git push

echo =====================================================================================
echo cdx_tf pip, conda, git done
echo =====================================================================================



