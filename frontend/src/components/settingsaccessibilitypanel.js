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

class SettingsAnalysisPanel extends React.Component {
    render() {
        const {
            updateSetting,
            updateValue
        } = this.props

        return (
            <Grid divided='vertically'>
                <Grid.Row columns={2}>
                    <Grid.Column>
                        <Checkbox
                            toggle
                            label='Invert colours'
                            onChange={(e, { checked }) => {
                                updateSetting('inverse', checked)
                                updateValue('inverse', checked)
                            }}
                            checked={this.props.inverse}
                        />
                    </Grid.Column>
                    <Grid.Column>
                        <Checkbox
                            toggle
                            label='Large text'
                            onChange={(e, { checked }) => {
                                //updateSetting('inverse', checked)
                                //updateValue('inverse', checked)
                            }}
                        />
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
)(SettingsAnalysisPanel)
