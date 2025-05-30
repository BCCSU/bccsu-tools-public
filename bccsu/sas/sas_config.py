# The java I need
# https://www.azul.com/downloads/?version=java-8-lts&os=windows&architecture=x86-64-bit&package=jdk#zulu
import os

SAS_config_names = ["autogen_winlocal"]

cpW = ("C:\\Program Files\\SASHome\\SASDeploymentManager\\9.4\\products\\deploywiz__94550__prt__xx__sp0__1\\deploywiz"
	   "\\sas.security.sspi.jar")
cpW += (";C:\\Program Files\\SASHome\\SASDeploymentManager\\9.4\\products\\deploywiz__94550__prt__xx__sp0__1"
		"\\deploywiz\\sas.svc.connection.jar")
cpW += (";C:\\Program Files\\SASHome\\SASDeploymentManager\\9.4\\products\\deploywiz__94550__prt__xx__sp0__1"
		"\\deploywiz\\log4j.jar")
cpW += (";C:\\Program Files\\SASHome\\SASDeploymentManager\\9.4\\products\\deploywiz__94550__prt__xx__sp0__1"
		"\\deploywiz\\sas.core.jar")
cpW += ";C:\\Users\\camer\\.virtualenvs\\bccsu\\Lib\\site-packages\\saspy\\java\\saspyiom.jar"


autogen_winlocal = {
    "java": r"C:\Program Files\Zulu\zulu-8\bin\java.exe",
    "encoding": "windows-1252",
    "classpath": cpW
}



os.environ["PATH"] += ";C:\\Program Files\\SASHome\\SASFoundation\\9.4\\core\\sasext"
