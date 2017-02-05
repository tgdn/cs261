# Copyright 2012-2013 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


DOC_EVENTS = {
    'doc-breadcrumbs': '.%s.%s',
    'doc-title': '.%s.%s',
    'doc-description': '.%s.%s',
    'doc-synopsis-start': '.%s.%s',
    'doc-synopsis-option': '.%s.%s.%s',
    'doc-synopsis-end': '.%s.%s',
    'doc-options-start': '.%s.%s',
    'doc-option': '.%s.%s.%s',
    'doc-option-example': '.%s.%s.%s',
    'doc-options-end': '.%s.%s',
    'doc-examples': '.%s.%s',
    'doc-output': '.%s.%s',
    'doc-subitems-start': '.%s.%s',
    'doc-subitem': '.%s.%s.%s',
    'doc-subitems-end': '.%s.%s',
    }


def fire_event(session, event_name, *fmtargs, **kwargs):
    event = session.create_event(event_name, *fmtargs)
    session.emit(event, **kwargs)


def generate_events(session, help_command):
    # First, register all events
    for event_name in DOC_EVENTS:
        session.register_event(event_name,
                               DOC_EVENTS[event_name])
    # Now generate the documentation events
    fire_event(session, 'doc-breadcrumbs', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-title', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-description', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-synopsis-start', help_command.event_class,
               help_command.name, help_command=help_command)
    if help_command.arg_table:
        for arg_name in help_command.arg_table:
            # An argument can set an '_UNDOCUMENTED' attribute
            # to True to indicate a parameter that exists
            # but shouldn't be documented.  This can be used
            # for backwards compatibility of deprecated arguments.
            if getattr(help_command.arg_table[arg_name],
                       '_UNDOCUMENTED', False):
                continue
            fire_event(session, 'doc-synopsis-option',
                       help_command.event_class,
                       help_command.name, arg_name,
                       arg_name=arg_name, help_command=help_command)
    fire_event(session, 'doc-synopsis-end', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-options-start', help_command.event_class,
               help_command.name, help_command=help_command)
    if help_command.arg_table:
        for arg_name in help_command.arg_table:
            if getattr(help_command.arg_table[arg_name],
                       '_UNDOCUMENTED', False):
                continue
            fire_event(session, 'doc-option', help_command.event_class,
                       help_command.name, arg_name,
                       arg_name=arg_name, help_command=help_command)
            fire_event(session, 'doc-option-example',
                       help_command.event_class,
                       help_command.name, arg_name,
                       arg_name=arg_name, help_command=help_command)
    fire_event(session, 'doc-options-end', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-subitems-start', help_command.event_class,
               help_command.name, help_command=help_command)
    if help_command.command_table:
        for command_name in sorted(help_command.command_table.keys()):
            if hasattr(help_command.command_table[command_name],
                       '_UNDOCUMENTED'):
                continue
            fire_event(session, 'doc-subitem', help_command.event_class,
                       help_command.name, command_name,
                       command_name=command_name, help_command=help_command)
    fire_event(session, 'doc-subitems-end', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-examples', help_command.event_class,
               help_command.name, help_command=help_command)
    fire_event(session, 'doc-output', help_command.event_class,
               help_command.name, help_command=help_command)
