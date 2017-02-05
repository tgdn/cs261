import React from 'react'

export default ({ trade }) => (
    <div>
        {trade.price} - {trade.sector} - {trade.symbol}
    </div>
)
