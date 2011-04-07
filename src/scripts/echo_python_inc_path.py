try:
    import distutils.sysconfig

    print distutils.sysconfig.get_python_inc()
except:
    raise SystemExit(1)

