# Database Configuration

Here are the steps in order to setup the database (Postgres).
You need to install postgresql database on your machine. I recommend
the [Mac app - PostgresApp](http://postgresapp.com/) as it works well,
easy to use and easy to install.

### Graphical client

If I were you I would also get [Postico](https://eggerapps.at/postico/)
which is a client for Mac. You can graphically see tables, rows, insert, delete,
update etc... Obviously that can be done on `psql` but when you want to simply
scroll down some data value and update it quickly graphically
[Postico](https://eggerapps.at/postico/) works pretty well.

So download that and set it up. Launch it and once that's done click on the
icon on the top bar and click on `psql`.
That will launch a Terminal inside an interactive `psql shell`.

You can type SQL in here, or use the psql commands: (there are many more)

| Command   | Description            |
| --------- | -----------------------|
| `\dt[+]`  | List tables in db      |
| `\l[+]`   | List databases         |
| `\dg`     | List roles             |
| `\?`      | See available commands |

The `[+]` means you can get more info if you add `+` at the end of the command.
For example `\l` will list databases, but `\l+` will list databases and more info
such as the size of the database.

## Create database

All commands below are to be entered in the interactive `psql` shell.
Type each line one after the other.

```sql

CREATE DATABASE cs261;
CREATE USER purple WITH PASSWORD 'purple123';
GRANT ALL PRIVILEGES ON DATABASE cs261 TO purple;
```

Right, now we have the database and the user setup and we are all on the same page.
