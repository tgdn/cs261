import React from 'react'
import { connect } from 'react-redux'
import { Grid, Button, Modal, Header, Icon, Input, Form } from 'semantic-ui-react'
import UploadButton from './uploadbutton'

class AnalysisModal extends React.Component {
    state = {
        opened: false,
        sendingStream: false,
    }

    constructor(props) {
        super(props)
        this.handleOpen = this.handleOpen.bind(this)
        this.handleClose = this.handleClose.bind(this)
        this.sendStream = this.sendStream.bind(this)
        this.handleUploadSuccess = this.handleUploadSuccess.bind(this)
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

    handleUploadSuccess(res) {
        const { filename } = res
        this.props.updateFile(res)
        this.props.notificationsystem.addNotification({
            allowHtml: true,
            level: 'success',
            children: (
                <div>
                    <strong>{filename}</strong> was uploaded successfully
                </div>
            )
        })
    }

    sendStream() {
        const { streamUrl, port } = this.props

        if (streamUrl && port) {
            this.setState({ sendingStream: true, })
            /* eslint-disable no-undef */
            fetch('/setstream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    streamUrl: this.props.streamUrl,
                    port: this.props.port || 80
                })
            })
            .then((res) => {
                if (res.status >= 200 && res.status < 300) {
                    return res.json()
                }
                const err = new Error(res.statusText)
                err.response = res
                throw err
            }).then(() => {
                this.setState({ sendingStream: false, })
            }).catch(() => {
                console.log('Request failed')
            })
            /* eslint-enable */
        }
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
                closeIcon
            >
                <Header icon='disk outline' content='Analyse trades' />
                <Modal.Content>
                    <Grid divided='vertically'>
                        <Grid.Row columns={2}>
                            <Grid.Column>
                                <Header size='medium'>Static file</Header>
                                <UploadButton
                                    uploadTo='/upload'
                                    handleSuccess={this.handleUploadSuccess}
                                    name='file'
                                    acceptedTypes={['text/csv']}
                                >
                                    <Icon name='cloud upload' />
                                    Select file...
                                </UploadButton>
                            </Grid.Column>
                            <Grid.Column>
                                <Header size='medium'>Live stream</Header>
                                <Form as='div'>
                                    <Form.Field>
                                        <Input
                                            placeholder='Stream url'
                                            onChange={this.props.updateStream}
                                            value={this.props.streamUrl}
                                        />
                                    </Form.Field>
                                    <Form.Field>
                                        <Input
                                            placeholder='Port'
                                            onChange={this.props.updatePort}
                                            value={this.props.port}
                                        />
                                    </Form.Field>
                                    <Button
                                        secondary
                                        type='button'
                                        onClick={this.sendStream}
                                    >
                                        <Icon name='laptop' /> Start analysing stream
                                    </Button>
                                </Form>
                            </Grid.Column>
                        </Grid.Row>
                    </Grid>
                </Modal.Content>
            </Modal>
        )
    }
}

AnalysisModal.propTypes = {
    streamUrl: React.PropTypes.string,
    port: React.PropTypes.number,
    file: React.PropTypes.string,
    updateFile: React.PropTypes.func,
    updateStream: React.PropTypes.func,
    updatePort: React.PropTypes.func,
}

AnalysisModal.defaultProps = {
    streamUrl: '',
    port: 80,
    file: '',
    updateFile: () => {},
    updateStream: () => {},
    updatePort: () => {},
}

export default connect(
    state => ({
        file: state.analysis.file,
        streamUrl: state.analysis.streamUrl,
        port: state.analysis.port,
        notificationsystem: state.notifications.notificationsystem,
    }),
    dispatch => ({
        updateFile: (res) => {
            console.log(res)
            // dispatch({
            //     type: 'UPDATE_FILE',
            //     data: {
            //         file: res.,
            //     }
            // })
        },
        updateStream: (e) => {
            dispatch({
                type: 'UPDATE_STREAM',
                data: {
                    streamUrl: e.target.value,
                }
            })
        },
        updatePort: (e) => {
            dispatch({
                type: 'UPDATE_PORT',
                data: {
                    port: e.target.value,
                }
            })
        }
    })
)(AnalysisModal)
