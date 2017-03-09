import React, { PropTypes } from 'react'
import {
    Container,
    Grid,
    Icon,
    Menu,
    Segment,
    Loader,
    Button,
    Label
} from 'semantic-ui-react'

import SymbolChart from './symbolchart'

const severityToColor = (severity) => {
    switch (severity) {
    case 2: return 'yellow'
    case 3: return 'blue'
    case 1:
    default:
        return 'red'
    }
}

const AlertDashboard = ({ alert, trades }) => (
    <div>
        <Menu size='huge' borderless attached='top' inverted>
            <Menu.Item header>
                <Icon name='warning sign' />
                {alert.description}
                <Label color={severityToColor(alert.severity)}>{alert.severity}</Label>
            </Menu.Item>
            <Menu.Menu position='right'>
                <Menu.Item as={Button}>
                    Review
                </Menu.Item>
                <Menu.Item as={Button}>
                    Not an anomaly
                </Menu.Item>
            </Menu.Menu>
        </Menu>
        <div>
            <Segment inverted attached='bottom'>
                {(trades.length === 0) ? (
                    <h1>Loading trades...</h1>
                ) : (
                    <SymbolChart
                        symbol='random'
                        trades={trades}
                        flagAnomalies={false}
                        flagOne={alert.trade_pk}
                    />
                )}
            </Segment>
        </div>
    </div>
)

export default AlertDashboard
