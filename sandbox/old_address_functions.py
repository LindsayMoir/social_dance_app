
    def try_google_search_for_postal(self, location):
        """
        Searches for a location using Google Maps Geocoding API and returns
        a postal code and the full formatted address, if available.

        Args:
            location (str): The location to search for.

        Returns:
            tuple: (postal_code, full_address) where postal_code is a string
                   (or None if not found) and full_address is the formatted address.
        """
        try:
            geocode_result = self.gmaps.geocode(location)
            if not geocode_result:
                return None, None

            # Take the first result as the best match.
            result = geocode_result[0]
            full_address = result.get("formatted_address")
            postal_code = None

            # Iterate over the address components to find the postal code.
            for component in result.get("address_components", []):
                if "postal_code" in component.get("types", []):
                    postal_code = component.get("long_name")
                    break

            return postal_code, full_address

        except Exception as e:
            # Log error or handle it as needed
            print(f"Error during geocoding: {e}")
            return None, None
        

    def query_geocoding_api(self, address, api_key):
        """
        Queries the Google Geocoding API and returns the JSON response.
        """
        endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": api_key}
        response = requests.get(endpoint, params=params)
        if response.status_code != 200:
            logging.warning(f"Geocoding API request failed (status {response.status_code}) for '{address}'")
            return None
        data = response.json()
        if data.get("status") != "OK":
            logging.warning(f"Geocoding API rejected: {data.get('status')} for '{address}'")
            return None
        return data
    

    def get_postal_code(self, address, api_key):
        """
        Extracts the postal code from the Google Geocoding API results.
        """
        data = self.query_geocoding_api(address, api_key)
        if data:
            for result in data.get("results", []):
                for component in result.get("address_components", []):
                    if "postal_code" in component.get("types", []):
                        return component.get("long_name")
        return None
    

    def get_municipality(self, address, api_key):
        """
        Extracts the municipality (locality) from the Google Geocoding API results.
        """
        data = self.query_geocoding_api(address, api_key)
        if data:
            for result in data.get("results", []):
                for component in result.get("address_components", []):
                    if "locality" in component.get("types", []):
                        return component.get("long_name")
        return None

    # ────────────────────────────────────────────────────────────────
    # Address Parsing and Building
    # ────────────────────────────────────────────────────────────────

    def parse_address_components(self, address):
        """
        Uses regex with named groups to parse a full address string.
        Expected format: "123 Main St, Toronto, ON M5J 2N1"
        """
        provinces = "ON|QC|NS|NB|MB|BC|PE|SK|AB|NL"
        pattern = (
            rf"(?P<street_number>\d+)\s+"
            rf"(?P<street_name>[A-Za-z0-9\s\.\-]+?)\s*"
            rf"(?P<street_type>St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive)?,\s+"
            rf"(?P<city>[A-Za-z\s\.\-]+),\s+"
            rf"(?P<province>{provinces})\s+"
            rf"(?P<postal_code>[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d)"
        )
        m = re.search(pattern, address)
        if m:
            return m.groupdict()
        return {}

    def build_address_dict_from_full(self, full_address):
        """
        Builds an address dictionary by parsing a full address string.
        """
        components = self.parse_address_components(full_address)
        return {
            "full_address": full_address,
            "street_number": components.get("street_number", ""),
            "street_name": components.get("street_name", ""),
            "street_type": components.get("street_type", ""),
            "postal_box": "",
            "city": components.get("city", ""),
            "province_or_state": components.get("province", ""),
            "postal_code": components.get("postal_code", ""),
            "country_id": "CA",
            "time_stamp": datetime.now()
        }

    def build_address_dict_from_db_row(self, row):
        """
        Constructs a formatted address and its dictionary from a DB row.
        """
        updated_location = self.format_address_from_db_row(row)
        address_dict = {
            "full_address": updated_location,
            "street_number": str(row.civic_no) if row.civic_no else "",
            "street_name": row.official_street_name or "",
            "street_type": row.official_street_type or "",
            "postal_box": "",
            "city": row.mail_mun_name or "",
            "province_or_state": row.mail_prov_abvn or "",
            "postal_code": row.mail_postal_code or "",
            "country_id": "CA",
            "time_stamp": datetime.now()
        }
        return updated_location, address_dict

    def extract_postal_and_address_from_candidate(self, candidate_text):
        """
        Searches candidate text (e.g., a concatenation of title and snippet)
        for a postal code, extracts a snippet around it, and calls
        extract_and_validate_address() to validate and extract a full address.
        """
        postal_match = re.search(r"[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d", candidate_text)
        if postal_match:
            postal_code = postal_match.group(0)
            start_index = max(0, postal_match.start() - 150)
            text_string = candidate_text[start_index: postal_match.end()]
            full_address, is_valid = self.extract_and_validate_address(text_string)
            if is_valid:
                return postal_code, full_address
        return None, None


    # ────────────────────────────────────────────────────────────────
    # DataFrame Address Cleanup Methods
    # ────────────────────────────────────────────────────────────────

    def clean_up_address(self, events_df):
        """
        Processes the events DataFrame to standardize the 'location' field and
        associate an address_id. It proceeds through these steps:
            1) Extract postal code via regex.
            2) If none, query Google API.
            3) If postal code exists, use DB lookup/fallback.
            4) Else, try Google search to extract postal code and address.
            5) Finally, if still no postal code, fallback using municipality.
        """
        logging.info(
            "clean_up_address(): Starting with events_df shape: %s", events_df.shape
        )
        address_id = 0
        updated_location = None

        for index, row in events_df.iterrows():
            raw_location = row.get('location')
            if not raw_location or pd.isna(raw_location):
                continue

            location = str(raw_location).strip()
            logging.info(
                "Processing location '%s' (event_id: %s)",
                location, row.get('event_id')
            )

            # Do we have a good address?
            address, is_valid = self.extract_and_validate_address(location)
            if is_valid:
                # 1. If the address is valid, update the address table.
                events_df = self.update_events_df(index, events_df, location, address)

            else:
                # 2. Attempt to extract postal code via regex.
                postal_code = self.extract_canadian_postal_code(location)
                logging.info(
                    "def clean_up_address(): Step #2 postal code %s from %s",
                    postal_code, location
                )

                if not postal_code:
                    address, is_valid = self.google_search_page_for_address(location)
                    if is_valid:
                        events_df = self.update_events_df(index, events_df, location, address)
                        logging.info(
                            "def clean_up_address(): Step #3 Valid address '%s' for '%s'",
                            address, location)
                          
                else:
                    # 4. Otherwise, try Google search with just location.
                    postal_code, updated_location = self.try_google_search_for_postal(
                        location
                    )
                    if postal_code and updated_location:
                        updated_location, address_id = self.populate_from_db_or_fallback(
                            location, postal_code
                        )
                        logging.info(f"def clean_up_address(): Step #4 Google search returned postal code: {postal_code}, "
                                    f"original location: {location}, "
                                    f"updated_location: {updated_location}")

                        # Do we have a valid address?
                        address, is_valid = self.extract_and_validate_address(location)
                        if is_valid:
                            events_df = self.update_events_df(index, events_df, location, address
                            )
                            logging.info(
                                "def clean_up_address(): Step #4 Google search returned "
                                "postal code: %s, original location: %s, updated_location: %s",
                                postal_code, location, updated_location)
                            
                        else:
                            # 5. Fallback to municipality if no postal code was found.
                            updated_location, address_id = self.fallback_with_municipality(
                                location
                            )
                            logging.info(
                                "def clean_up_address(): Step #5 fallback_with_municipality "
                                "returned address_id: %s, original location: %s, "
                                "updated_location: %s",
                                address_id, location, updated_location
                            )

                            # Do we have a valid address?
                            address, is_valid = self.extract_and_validate_address(location)
                            if is_valid:
                                events_df = self.update_events_df(
                                    index, events_df, location, address
                                )
                                logging.info(
                                    "def clean_up_address(): Step #5 Google search returned "
                                    "postal code: %s, original location: %s, updated_location: %s",
                                    postal_code, location, updated_location
                                )
                            else:
                                # Insert municipality into the original location.
                                events_df.loc[index, 'location'] = location + updated_location

        return events_df


    # ────────────────────────────────────────────────────────────────
    # Additional Helper Methods
    # ────────────────────────────────────────────────────────────────

    def google_search_page_for_address(self, query):
        """
        Performs a Google search by constructing the search URL, scrapes the
        resulting Google search page, and extracts a Canadian address.

        Steps:
        1. Build a Google search URL using the query.
        2. Use Playwright to load the search page and extract its text.
        3. Find the first occurrence of a Canadian postal code in the text.
        4. Extract 150 characters preceding the postal code (plus the postal code itself).
        5. Pass this snippet to extract_and_validate_address() to get the full address.

        Args:
            query (str): The partial address or search term.

        Returns:
            tuple: (address, is_valid) extracted from the page, or (None, False) if not found.
        """
        # Build the Google search URL.
        base_url = "https://www.google.com/search?q="
        search_url = base_url + quote(query)
        logging.info("google_search_page_for_address: Searching with URL: %s", search_url)
        
        # Extract text content from the Google search page.
        text_content = read_extract.extract_text_with_playwright(search_url)
        logging.info(f"google_search_page_for_address: Extracted text content from URL: {search_url}, \n{text_content}")
        if not text_content:
            logging.error("google_search_page_for_address: Failed to extract text from URL: %s", search_url)
            return None, False

        # Find the first Canadian postal code in the page text.
        postal_code_match = re.search(r"[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d", text_content)
        if not postal_code_match:
            logging.info("google_search_page_for_address: No Canadian postal code found in search results.")
            return None, False

        # Extract 150 characters before the postal code (or from start if not enough characters)
        start_index = max(0, postal_code_match.start() - 150)
        text_snippet = text_content[start_index: postal_code_match.end()]
        logging.info("google_search_page_for_address: Extracted text snippet: %s", text_snippet)

        # Extract the full address using your provided regex function.
        address, is_valid = self.extract_and_validate_address(text_snippet)
        return address, is_valid
    

    def update_events_df(self, index, events_df, location, updated_location):
        """
        Update the events DataFrame with a new location and corresponding address ID.
        Parameters:
        index (int): The index of the event to update in the DataFrame.
        events_df (pd.DataFrame): The DataFrame containing event information.
        location (str): The original location of the event.
        updated_location (str): The new location to update in the DataFrame.
        Returns:
        pd.DataFrame: The updated DataFrame with the new location and address ID.
        """
        address_dict = self.build_address_dict_from_full(updated_location)
        address_id = db_handler.get_address_id(address_dict)
        logging.info(f"def clean_up_address(): Step #1 Valid address '{updated_location}' for '{location}'")

        address_dict = self.build_address_dict_from_full(updated_location)
        address_id = db_handler.get_address_id(address_dict)
        logging.info(f"def clean_up_address(): Step #1 Valid address '{updated_location}' for '{location}'")

        # Update the DataFrame
        if updated_location:
            events_df.loc[index, 'location'] = updated_location
        events_df.loc[index, 'address_id'] = address_id
        logging.info(f"Updated location '{updated_location}' with address_id: {address_id}")

        return events_df
    
    
    def extract_building_name(self, location):
        """
        Extracts the building name from the location string.
        The building name is defined as the initial sequence of characters 
        that does not include any numeric digits.

        Args:
            location (str): The location string that may include a building name.

        Returns:
            str or None: The extracted building name, or None if no non-numeric prefix exists.
        """
        match = re.match(r'^([^0-9]+)', location)
        if match:
            # Strip any extra whitespace from the extracted building name.
            building_name = match.group(1).strip()
            return building_name
        return None


    def extract_canadian_postal_code(self, location_str):
        """
        Extracts a Canadian postal code from a location string using regex.
        """
        match = re.search(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d', location_str)
        if match:
            possible_pc = match.group().replace(' ', '')
            if self.is_canadian_postal_code(possible_pc):
                return possible_pc
        return None
    

    def populate_from_db_or_fallback(self, location_str, postal_code):
        """
        Given a valid postal code, attempts to find a matching address in the DB.
        If no match is found, falls back by using municipality information.
        Returns a tuple (updated_location, address_id).
        """
        if location_str:
            numbers = re.findall(r'\d+', location_str)
        query = """
            SELECT
                civic_no,
                civic_no_suffix,
                official_street_name,
                official_street_type,
                official_street_dir,
                mail_mun_name,
                mail_prov_abvn,
                mail_postal_code
            FROM locations
            WHERE mail_postal_code = %s;
            """
        
        df = pd.read_sql(query, db_handler.address_db_engine, params=(postal_code,))
        if df.empty:
            # No DB match; fallback via municipality.
            municipality = self.get_municipality(location_str, self.google_api_key)
            updated_location = f"{location_str}, {municipality}, BC, {postal_code}, CA"
            updated_location = updated_location.replace('None,', '').strip()
            logging.info("No DB match for postal code '%s'. Fallback location: '%s'", postal_code, updated_location)
            address_dict = self.build_address_dict_from_full(updated_location)
            address_id = db_handler.get_address_id(address_dict)
            return updated_location, address_id

        # If multiple rows, try to match a civic number.
        if numbers:
            row = df.iloc[0] if df.shape[0] == 1 else df.loc[self.match_civic_number(df, numbers)]
            updated_location, address_dict = self.build_address_dict_from_db_row(row)
            address_id = db_handler.get_address_id(address_dict)
            logging.info("DB match for postal code '%s': '%s'", postal_code, updated_location)
            return updated_location, address_id
        else:
            return None, None
    

    def fallback_with_municipality(self, location_str):
        """
        If no postal code is found, attempts to obtain municipality information via Google.
        If the municipality is recognized (from a provided list), returns a partial location and address_id.
        """
        municipality = self.get_municipality(location_str, self.google_api_key)
        if not municipality:
            return None, None

        with open(self.config['input']['municipalities'], 'r', encoding='utf-8') as f:
            muni_list = [line.strip() for line in f if line.strip()]
        if municipality in muni_list:
            updated_location = f"{location_str}, {municipality}, BC, CA"
            updated_location = updated_location.replace('None', '').replace(',,', ',').strip()
            address_dict = self.build_address_dict_from_full(updated_location)
            address_id = db_handler.get_address_id(address_dict)
            logging.info("Fallback using municipality: '%s'", updated_location)
            return updated_location, address_id
        return None, None
    

    def match_civic_number(self, df, numbers):
        """
        Matches the first number found in the location string to a civic_no in the DataFrame.
        """
        if not numbers:
            return df.index[0]
        for i, addr_row in df.iterrows():
            if addr_row.civic_no is not None:
                try:
                    if int(numbers[0]) == int(addr_row.civic_no):
                        return i
                except ValueError:
                    continue
        return df.index[0]
    

    def format_address_from_db_row(self, db_row):
        """
        Constructs a formatted address string from a DB row.
        """
        parts = [
            str(db_row.civic_no) if db_row.civic_no else "",
            str(db_row.civic_no_suffix) if db_row.civic_no_suffix else "",
            db_row.official_street_name or "",
            db_row.official_street_type or "",
            db_row.official_street_dir or ""
        ]
        street_address = " ".join(part for part in parts if part).strip()
        city = db_row.mail_mun_name or ""
        formatted = f"{street_address}, {city}, {db_row.mail_prov_abvn or ''}, {db_row.mail_postal_code or ''}, CA"
        formatted = re.sub(r'\s+,', ',', formatted)
        formatted = re.sub(r',\s+,', ',', formatted)
        return re.sub(r'\s+', ' ', formatted).strip()


    def is_canadian_postal_code(self, postal_code):
        """
        Validates if a string matches the Canadian postal code format (A1A 1A1).
        """
        pattern = r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$'
        return bool(re.match(pattern, postal_code.strip()))
    
    
    def extract_and_validate_address(self, text_string):
        """
        Extracts a full Canadian address from a given text string and validates its construction.
        
        The expected format is:
        [street number] [street name], [city], [province] [postal code]
        
        Where:
        - Province is one of the common two-letter abbreviations: ON, QC, NS, NB, MB, BC, PE, SK, AB, NL.
        - Postal code follows the pattern: A1A 1A1 (with or without a space).
        
        Parameters:
        text_string (str): The text containing the address.
        
        Returns:
        tuple: (address, is_valid) where 'address' is the extracted address (or None if not found)
                and 'is_valid' is a boolean indicating if a properly constructed address was found.
        """
        
        # Define a regex for a Canadian postal code: letter-digit-letter [optional space] digit-letter-digit
        postal_code_pattern = r"[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d"
        
        # List of valid province abbreviations
        provinces = "ON|QC|NS|NB|MB|BC|PE|SK|AB|NL"
        
        # Build a regex pattern for a full address.
        # This pattern assumes the address is like:
        #   "123 Main St, Toronto, ON M5J 2N1"
        address_regex = (
            r"(\d+\s+[A-Za-z0-9\s\.\-]+,\s+"  # street number and name
            r"[A-Za-z\s\.\-]+,\s+"             # city name
            r"(?:{provinces})\s+"             # province abbreviation
            + postal_code_pattern + ")"       # postal code
        ).format(provinces=provinces)
        
        # Search for the address pattern in the text string
        match = re.search(address_regex, text_string)
        
        if match:
            address = match.group(1)
            # If we found a match, we assume it is a properly constructed address.
            # Further checks could include verifying individual parts or using a dedicated address parser.
            is_valid = True
        else:
            address = None
            is_valid = False

        return address, is_valid
