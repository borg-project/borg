import sys
import datetime

APPNAME="borg"
VERSION=datetime.date.today().strftime("%y.%m.%d")

top = "."
out = "build"

def options(context):
    context.parser.set_defaults(prefix = sys.prefix)

    context.recurse("src")

def configure(context):
    context.recurse("src")

def build(context):
    context.recurse("src")

