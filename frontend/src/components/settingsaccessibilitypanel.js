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

const SettingsAnalysisPanel = ({
    updateSetting, updateValue, inverse, largetext
}) => (
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
                    checked={inverse}
                />
            </Grid.Column>
            <Grid.Column>
                <Checkbox
                    toggle
                    label='Large text'
                    onChange={(e, { checked }) => {
                        updateSetting('largetext', checked)
                        updateValue('largetext', checked)
                    }}
                    checked={largetext}
                />
            </Grid.Column>
        </Grid.Row>
    </Grid>
)

export default connect(
    state => ({
        settings: state.db.settings,
        inverse: state.settings.inverse,
        largetext: state.settings.largetext
    }),
)(SettingsAnalysisPanel)
