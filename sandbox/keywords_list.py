keywords = ["2-step", "cha cha", "double shuffle", "country waltz", "line dance", "night club", "nite club", "nite club 2", 
            "nite club two", "two step", "west coast swing", "wcs", "kizomba", "urban kiz", "semba", "tarraxo", "tarraxa", 
            "tarraxinha", "douceur", "salsa", "bachata", "kizomba", "merengue", "cha cha cha", "swing", "balboa", "lindy hop", 
            "east coast swing", "swing", "balboa", "lindy hop", "east coast swing", "milonga", "tango", "west coast swing", "wcs"]

keywords = set(keywords)
keywords = list(keywords)
keywords.sort()

# Take the list and turn it into a single string with the elements separated by commas
keywords_str = ', '.join(keywords)
print(keywords_str)
