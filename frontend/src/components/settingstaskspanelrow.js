import React from 'react'
import { connect } from 'react-redux'
import {
    Icon,
    Table
} from 'semantic-ui-react'

const SettingsTasksPanelRow = ({ task, killProcess, notificationsystem }) => (
    <Table.Row>
        <Table.Cell>
            <Icon name='disk outline' />
            {`${task.task.charAt(0).toUpperCase()}${task.task.slice(1)}`}
        </Table.Cell>
        <Table.Cell>{task.type}</Table.Cell>
        <Table.Cell>{task.pid}</Table.Cell>
        <Table.Cell>{task.created_at.toUTCString()}</Table.Cell>
        <Table.Cell selectable collapsing textAlign='center'>
            <a
                href=''
                onClick={(e) => {
                    e.preventDefault()
                    killProcess(task.id, notificationsystem)
                }}
            >
                <Icon name='remove circle' />
                Kill task
            </a>
        </Table.Cell>
    </Table.Row>
)

export default connect(
    state => ({
        notificationsystem: state.notifications.notificationsystem,
    }),
)(SettingsTasksPanelRow)
