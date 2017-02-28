import React from 'react'
import { connect } from 'react-redux'
import { Table } from 'semantic-ui-react'

import SettingsTasksPanelRow from './settingstaskspanelrow'

const killProcess = (id, notificationsystem) => {
    /* eslint-disable no-undef */
    fetch('/killprocess', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id })
    })
    .then((res) => {
        if (res.status >= 200 && res.status < 300) {
            return res.json()
        }
        const err = new Error(res.statusText)
        err.response = res
        throw err
    }).then((res) => {
        console.log(res);
    }).catch(() => {
        notificationsystem.addNotification({
            level: 'error',
            message: 'An error occurred while processing your request, please try again'
        })
    })
    /* eslint-enable */
}

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
                <SettingsTasksPanelRow key={task.id} task={task} killProcess={killProcess} />
            ))}
        </Table.Body>
    </Table>
)

export default connect(
    state => ({
        tasks: state.tasks.tasks,
    }),
)(SettingsTasksPanel)
