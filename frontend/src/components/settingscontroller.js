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

import SettingsAnalysisPanel from './settingsanalysispanel'

class SettingsController extends React.Component {

    panelStyle = { margin: '0 0 1.2rem' }

    constructor(props) {
        super(props)
        this.onLiveBtnClick = this.onLiveBtnClick.bind(this)
        this.onStaticBtnClick = this.onStaticBtnClick.bind(this)
    }

    updateValue(k,v) {
        this.props.settings.upsert({
            key: [k],
            value: [v]
        }).subscribe()
    }

    onStaticBtnClick(e) {
        if (this.props.mode == 'live') {
            this.props.updateSetting('mode', 'static')
        }
    }

    onLiveBtnClick(e) {
        if (this.props.mode == 'static') {
            this.props.updateSetting('mode', 'live')
        }
    }

    render() {

        return (
            <Grid>
                <Grid.Row columns={1}>
                    <Grid.Column>
                        <div style={this.panelStyle}>
                            <Header size='small'>Analysis</Header>
                            <Container textAlign='center'>
                                <Button.Group basic>
                                    <Button
                                        toggle
                                        active={this.props.mode === 'static'}
                                        onClick={this.onStaticBtnClick}
                                    >
                                        Static file
                                    </Button>
                                    <Button
                                        toggle
                                        active={this.props.mode === 'live'}
                                        onClick={this.onLiveBtnClick}
                                    >
                                        Live stream
                                    </Button>
                                </Button.Group>
                            </Container>
                            <SettingsAnalysisPanel />
                        </div>
                        <div style={this.panelStyle}>
                            <Header size='small'>Accessibility</Header>
                            <Checkbox toggle label='Invert colours' />
                        </div>
                        <Divider />
                        <div style={this.panelStyle}>
                            <Header size='small' color='red'>Danger Zone</Header>
                            <Button color='red' icon='warning sign' content='Reset database' />
                        </div>
                    </Grid.Column>
                </Grid.Row>
            </Grid>
        )
    }
}

export default connect(
    state => ({
        settings: state.db.settings,
        mode: state.settings.mode,
        inverse: state.settings.inverse,
    }),
    dispatch => ({
        updateSetting: (k,v) => {
            dispatch({
                type: 'SET_SETTING',
                data: {
                    key: k,
                    value: v,
                }
            })
        }
    })
)(SettingsController)
