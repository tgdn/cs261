
const r = require('rethinkdb');
const pgp = require('pg-promise')({});

const dbConfig = {
    host: 'localhost',
    port: 5432,
    database: 'cs261',
    user: 'purple',
    password: 'purple123'
};
const db = pgp(dbConfig);

const tradeFields = 'id, price, bid, ask, size, flagged, datetime';

const handleException = (err, res, reason) => {
    console.error(err) // eslint-disable-line
    res.status(500)
        .json({ success: false, reason })
}

const getSymbols = (req, res) => {
    db.any('SELECT * FROM symbols ORDER BY name')
    .then((symbols) => {
        res.status(200)
            .json({
                success: true,
                symbols
            })
    })
    .catch(err => handleException(err, res))
}

const getSymbol = (req, res) => {
    const symbol = req.params.symbol || null
    if (symbol != null) {
        // Get latest 1000 trades
        // Nest query to reorder trades by datetime ASC
        db.any(
            `SELECT * FROM (
                SELECT $(tradeFields^)
                FROM trades WHERE symbol_name = $(symbol)
                ORDER BY datetime DESC
            ) AS derivedTable ORDER BY datetime ASC`,
        { tradeFields, symbol })
        .then((trades) => {
            res.status(200)
                .json({
                    success: true,
                    trades
                })
        })
        .catch(err => handleException(err, res))
    }
}

const getFlaggedTrades = (req, res) => {
    const tradeid = req.params.tradeid
    if (tradeid != null) {
        // try and get initial trade
        db.oneOrNone('SELECT id, symbol_name FROM trades WHERE id = $1', tradeid)
        .then((trade) => {
            if (trade != null) {
                // get some trades on each side of flagged trade
                db.any(
                    `SELECT * FROM (
                        (
                            SELECT $(tradeFields^) FROM trades
                            WHERE id <= $(tradeid) AND symbol_name = $(symbol_name)
                            ORDER BY datetime DESC LIMIT 201
                        )
                        UNION
                        (
                            SELECT $(tradeFields^) FROM trades
                            WHERE id > $(tradeid) AND symbol_name = $(symbol_name)
                            ORDER BY datetime ASC LIMIT 30
                        )
                    ) AS sbq ORDER BY datetime ASC`,
                    {
                        tradeFields,
                        tradeid: trade.id,
                        symbol_name: trade.symbol_name
                    }
                )
                .then((trades) => {
                    res.status(200)
                        .json({
                            success: true,
                            trades,
                        })
                })
            } else { /* trade does not exist */
                res.status(404)
                    .json({
                        success: true,
                        trades: []
                    })
            }
        })
        .catch(err => handleException(err, res))
    }
}

const getFlaggedTimeConstraintTrades = (req, res) => {
    const hour = req.params.hour
    const symbol = req.params.symbol
    if (hour && symbol) {
        db.any(
            `SELECT $(tradeFields^) FROM trades
            WHERE
                EXTRACT(hour from datetime) >= $(hour) - 1 AND
                EXTRACT(hour from datetime) < $(hour) + 2 AND
                symbol_name = $(symbol)
            ORDER BY datetime ASC`,
            { tradeFields, hour, symbol }
        )
        .then((trades) => {
            res.status(200).json({ success: true, trades })
        })
        .catch(err => handleException(err, res))
    }
}

const tradesBefore = (req, res) => {
    const before = req.params.before
    const symbol = req.params.symbol
    const count = req.params.count || 50
    if (before != null && symbol != null) { // eslint-disable-line
        db.any(
            `SELECT * FROM (
                SELECT $(tradeFields^)
                FROM trades
                WHERE id < $(before) AND symbol_name = $(symbol)
                ORDER BY datetime DESC LIMIT $(count)
            ) AS sbq ORDER BY datetime ASC`,
            { tradeFields, before, symbol, count }
        )
        .then((trades) => {
            res.status(200)
                .json({
                    success: true,
                    trades
                })
        })
        .catch(err => handleException(err, res))
    }
}

const tradesAfter = (req, res) => {
    const after = req.params.after
    const symbol = req.params.symbol
    const count = req.param.count || 50
    if (after != null && symbol != null) { // eslint-disable-line
        db.any(
            `SELECT $(tradeFields^)
            FROM trades
            WHERE id > $(after) AND symbol_name = $(symbol)
            ORDER BY datetime ASC LIMIT $(count)`,
            { tradeFields, after, symbol, count }
        )
        .then((trades) => {
            res.status(200)
                .json({
                    success: true,
                    trades
                })
        })
        .catch(err => handleException(err, res))
    }
}

const getTrade = (req, res) => {
    const tradeid = parseInt(req.body.tradeid, 10)
    db.one('SELECT $(tradeFields^) FROM trades WHERE id = $(tradeid)',
    { tradeFields, tradeid })
    .then((trade) => {
        res.status(200)
            .json({ trade })
    })
    .catch(err => handleException(err, res))
}

const searchAlerts = (req, res, conn) => {
    const rawTerm = req.body.term
    if (rawTerm) {
        const term = rawTerm.toString().toLowerCase()
        r.table('alerts')
        .filter(alert => alert('description').downcase().match(`(?i)${term}`))
        .orderBy('severity') // ascending is default
        .limit(50)
        .run(conn, (queryErr, cursor) => {
            if (!queryErr) {
                cursor.toArray((err, alerts) => {
                    if (!err) {
                        res.status(200).json({ alerts })
                        return // eslint-disable-line
                    }
                })
            } else {
                console.error(queryErr);
                res.status(200).json({ alerts: [] })
            }
        })
    } else {
        res.status(200).json({ alerts: [] })
    }
}

const cancelOneAlert = (req, res, conn) => {
    const alertid = req.body.alertid
    if (alertid) {
        r.table('alerts')
            .get(alertid)
            .delete({ returnChanges: true })
            .run(conn, (err, result) => {
                if (!err) {
                    // get tradeid from result set
                    const tradeid = result.changes[0].old_val.trade_pk
                    db.none(
                        'UPDATE trades SET flagged = FALSE WHERE id = $(tradeid)',
                        { tradeid }
                    )
                    .then(() => {
                        res.status(200).json({ success: true })
                    })
                    .catch((qerr) => {
                        console.log(qerr)
                        res.status(200).json({ success: false })
                    })
                } else {
                    console.error(err);
                    res.status(200).json({ success: false })
                }
            })
    } else {
        res.status(200).json({ success: false })
    }
}

const getAlertCount = (req, res, conn) => {
    r.table('alerts')
        .filter({ reviewed: false })
        .count()
        .run(conn, (err, count) => {
            if (!err) {
                res.status(200).json({ count })
            } else {
                console.log(err);
                res.status(200).json({ count: 0 })
            }
        })
}

module.exports = {
    getSymbols,
    getSymbol,
    getFlaggedTrades,
    getTrade,
    tradesBefore,
    tradesAfter,
    searchAlerts,
    cancelOneAlert,
    getAlertCount,
    getFlaggedTimeConstraintTrades,
}
