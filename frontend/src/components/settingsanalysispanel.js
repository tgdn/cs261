/* eslint-disable react/sort-comp */

import React from 'react'
import { connect } from 'react-redux'
import {
    Container,
    Grid,
    Button,
    Header,
    Icon,
    Input,
    Form,
    Checkbox,
    Label,
    Divider
} from 'semantic-ui-react'
import UploadButton from './uploadbutton'

class SettingsAnalysisPanel extends React.Component {
    liveStyle = {
        width: '40%',
        margin: '1rem auto',
        textAlign: 'center',
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

    renderLivePanel() {
        return (
            <Form as='div' style={this.liveStyle}>
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
        )
    }

    renderStaticPanel() {
        return (
            <UploadButton
                uploadTo='/upload'
                handleSuccess={this.handleUploadSuccess}
                name='file'
                acceptedTypes={['text/csv']}
            >
                <Icon name='cloud upload' />
                Select file...
            </UploadButton>
        )
    }

    render() {
        return this.props.mode === 'live' ? this.renderLivePanel() : this.renderStaticPanel()
    }
}

export default connect(
    state => ({
        mode: state.settings.mode,
        inverse: state.settings.inverse,
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
)(SettingsAnalysisPanel)
