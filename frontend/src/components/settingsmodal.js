/* eslint-disable react/sort-comp */

import React from 'react'
import { Button, Modal, Header, Icon } from 'semantic-ui-react'

import SettingsController from './settingscontroller'

class SettingsModal extends React.Component {
    state = {
        opened: false,
    }

    constructor(props) {
        super(props)
        this.handleOpen = this.handleOpen.bind(this)
        this.handleClose = this.handleClose.bind(this)
    }

    handleOpen() {
        this.setState({ opened: true, })
    }

    handleClose() {
        this.setState({ opened: false, })
    }

    render() {
        return (
            <Modal
                trigger={
                    <Button
                        inverted
                        icon='setting'
                        color='grey'
                        onClick={this.handleOpen}
                        circular
                    />
                }
                open={this.state.opened}
                onClose={this.handleClose}
                size='small'
                closeIcon
            >
                <Header icon='settings' content='Preferences' />
                <Modal.Content>
                    <SettingsController />
                </Modal.Content>
            </Modal>
        )
    }
}

export default SettingsModal
