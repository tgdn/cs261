import React from 'react'
import { connect } from 'react-redux'
import {
    Icon,
    Table
} from 'semantic-ui-react'

const SettingsTasksPanel = ({ tasks }) => (
    <Table compact='very' size='small' inverted selectable>
        <Table.Header>
            <Table.Row>
                <Table.HeaderCell>Task</Table.HeaderCell>
                <Table.HeaderCell>Type</Table.HeaderCell>
                <Table.HeaderCell>PID</Table.HeaderCell>
                <Table.HeaderCell>Created at</Table.HeaderCell>
                <Table.HeaderCell textAlign='center'>Action</Table.HeaderCell>
            </Table.Row>
        </Table.Header>
        <Table.Body>
            {tasks.map(task => (
                <Table.Row key={task.id}>
                    <Table.Cell>
                        <Icon name='disk outline' />
                        {`${task.task.charAt(0).toUpperCase()}${task.task.slice(1)}`}
                    </Table.Cell>
                    <Table.Cell>{task.type}</Table.Cell>
                    <Table.Cell>{task.pid}</Table.Cell>
                    <Table.Cell>{task.created_at.toUTCString()}</Table.Cell>
                    <Table.Cell selectable collapsing textAlign='center'>
                        <a href='#'>
                            <Icon name='remove circle' />
                            Kill task
                        </a>
                    </Table.Cell>
                </Table.Row>
            ))}
        </Table.Body>
    </Table>
)

export default connect(
    state => ({
        tasks: state.tasks.tasks,
    }),
)(SettingsTasksPanel)
