# 2025-04-07

* Allow Serena to switch between projects (project activation)
    * Add central Serena configuration in `serena_config.yml`, which 
        * contains the list of available projects
        * allows to configure whether project activation is enabled
        * now contains the GUI logging configuration (project configurations no longer do)
    * Add new tools `activate_project` and `get_active_project`
    * Providing a project configuration file in the launch parameters is now optional
* Logging:
    * Improve error reporting in case of initialization failure: 
      open a new GUI log window showing the error or ensure that the existing log window remains visible for some time
* Language servers:
    * Fix C# language server initialization issue when the project path contains spaces
* Agno: 
    * Fix Agno reloading mechanism causing failures when initializing the sqlite memory database #8
    * Fix Serena GUI log window not capturing logs after initialization

# 2025-04-01

Initial public version


