/* eslint-disable react/sort-comp, react/prop-types */

import React from 'react'
import { connect } from 'react-redux'
import {
    Container,
    Grid,
    Button,
    Header,
    Divider,
    Confirm,
    Dimmer,
    Loader
} from 'semantic-ui-react'

import SettingsAnalysisPanel from './settingsanalysispanel'
import SettingsAccessibilityPanel from './settingsaccessibilitypanel'
import SettingsTasksPanel from './settingstaskspanel'

class SettingsController extends React.Component {

    panelStyle = { margin: '0 0 1.2rem' }

    state = {
        resetDbConfirmOpened: false,
        resetingDb: false,
    }

    constructor(props) {
        super(props)
        this.updateValue = this.updateValue.bind(this)
        this.onLiveBtnClick = this.onLiveBtnClick.bind(this)
        this.onStaticBtnClick = this.onStaticBtnClick.bind(this)
        this.onResetBtnClick = this.onResetBtnClick.bind(this)
        this.handleResetConfirm = this.handleResetConfirm.bind(this)
        this.handleResetCancel = this.handleResetCancel.bind(this)
    }

    updateValue(k, v) {
        this.props.settings.upsert({
            id: k,
            value: v
        }).subscribe()
    }

    onStaticBtnClick() {
        if (this.props.mode === 'live') {
            this.props.updateSetting('mode', 'static')
            this.updateValue('mode', 'static')
        }
    }

    onLiveBtnClick() {
        if (this.props.mode === 'static') {
            this.props.updateSetting('mode', 'live')
            this.updateValue('mode', 'live')
        }
    }

    onResetBtnClick() {
        this.setState({ resetDbConfirmOpened: true, })
    }

    handleResetConfirm() {
        this.setState({
            resetDbConfirmOpened: false,
            resetingDb: true,
        })
        /* now disconnect from horizon */
        this.props.horizon.disconnect()

        /* eslint-disable no-undef */
        fetch('/resetdb', { method: 'POST', })
        .then((res) => {
            if (res.status >= 200 && res.status < 300) {
                return res.json()
            }
            const err = new Error(res.statusText)
            err.response = res
            throw err
        }).then(() => {
            this.props.notificationsystem.addNotification({
                level: 'success',
                message: 'The database was correctly reset'
            })
            /* refresh the page */
            setTimeout(() => { window.location = window.location.href }, 1000)
        }).catch(() => {
            this.props.notificationsystem.addNotification({
                level: 'error',
                message: 'An unexpected error occurred, please try again'
            })
        })
        /* eslint-enable */
    }

    handleResetCancel() {
        this.setState({ resetDbConfirmOpened: false, })
    }

    render() {
        return (
            <Grid>
                <Dimmer
                    active={this.state.resetingDb}
                    page
                >
                    <Loader size='large'>Reseting database...</Loader>
                </Dimmer>
                <Grid.Row columns={1}>
                    <Grid.Column>
                        <div style={this.panelStyle}>
                            <Container textAlign='center'>
                                <Button.Group basic>
                                    <Button
                                        active={this.props.mode === 'static'}
                                        onClick={this.onStaticBtnClick}
                                    >
                                        Static file
                                    </Button>
                                    <Button
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
                            <Header size='small'>Current tasks</Header>
                            <SettingsTasksPanel />
                        </div>
                        <div style={this.panelStyle}>
                            <Header size='small'>Accessibility</Header>
                            <SettingsAccessibilityPanel
                                updateSetting={this.props.updateSetting}
                                updateValue={this.updateValue}
                            />
                        </div>
                        <Divider />
                        <div style={this.panelStyle}>
                            <Header size='small' color='red'>Danger Zone</Header>
                            <Button
                                color='red'
                                icon='warning sign'
                                content='Reset database'
                                onClick={this.onResetBtnClick}
                            />
                            <Confirm
                                open={this.state.resetDbConfirmOpened}
                                onCancel={this.handleResetCancel}
                                onConfirm={this.handleResetConfirm}
                                header='Are you sure you want to reset the database?'
                                content='This action cannot be undone'
                                confirmButton='Yes'
                            />
                        </div>
                    </Grid.Column>
                </Grid.Row>
            </Grid>
        )
    }
}

export default connect(
    state => ({
        horizon: state.db.horizon,
        settings: state.db.settings,
        mode: state.settings.mode,
        inverse: state.settings.inverse,
        notificationsystem: state.notifications.notificationsystem,
    }),
    dispatch => ({
        updateSetting: (k, v) => {
            dispatch({
                type: 'SET_SETTING',
                data: {
                    id: k,
                    value: v,
                }
            })
        }
    })
)(SettingsController)
