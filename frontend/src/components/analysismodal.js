import React from 'react'
import { connect } from 'react-redux'
import { Grid, Button, Modal, Header, Icon, Input, Form } from 'semantic-ui-react'
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
                closeIcon={true}
            >
                <Header icon='terminal' content='Analyse trades' />
                <Modal.Content>
                    <Grid divided='vertically'>
                        <Grid.Row columns={2}>
                            <Grid.Column>
                                <Header size='medium'>Static file</Header>
                                <UploadButton
                                    uploadTo='/upload'
                                    handleSuccess={this.props.updateFile}
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
                                    <Button secondary type='button'>
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

export default connect(
    state => ({
        file: state.analysis.file,
        streamUrl: state.analysis.streamUrl,
        port: state.analysis.port
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
