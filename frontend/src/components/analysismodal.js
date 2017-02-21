import React from 'react'
import { Button, Modal, Header, Icon } from 'semantic-ui-react'
import UploadButton from './uploadbutton'

class AnalysisModal extends React.Component {
    state = {
        opened: false,
    }

    constructor(props) {
        super(props)
        this.handleOpen = this.handleOpen.bind(this)
        this.handleClose = this.handleClose.bind(this)
    }

    handleOpen() {
        this.setState({
            opened: true,
        })
    }

    handleClose() {
        this.setState({
            opened: false,
        })
    }

    render() {
        return (
            <Modal
                trigger={
                    <Button inverted color='grey' onClick={this.handleOpen}>
                        Analyse
                    </Button>
                }
                open={this.state.opened}
                onClose={this.handleClose}
                size='small'
            >
                <Header icon='terminal' content='Analyse trades' />
                <Modal.Content>
                    <p>Hello</p>
                    <button>Hello</button>
                    <UploadButton uploadTo='/upload'>
                        Upload
                    </UploadButton>
                </Modal.Content>
                <Modal.Actions>
                    <Button onClick={this.handleClose}>
                        <Icon name='remove' /> Cancel
                    </Button>
                    <Button secondary>
                        <Icon name='checkmark' /> Yes
                    </Button>
                </Modal.Actions>
            </Modal>
        )
    }
}

export default AnalysisModal
