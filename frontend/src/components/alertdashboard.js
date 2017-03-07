import React, { PropTypes } from 'react'
import {
    Container,
    Grid,
    Icon,
    Menu,
    Segment,
    Loader
} from 'semantic-ui-react'

import SymbolChart from './symbolchart'

const AlertDashboard = ({ alert, trades }) => (
    <div>
        <Menu size='huge' borderless attached='top' inverted>
            <Menu.Item header>
                <Icon name='warning sign' />
                {alert.description}
            </Menu.Item>
            <Menu.Menu position='right'>
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
