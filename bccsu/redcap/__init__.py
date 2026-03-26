from lxml import etree
import numpy as np
import pandas as pd
import re


def _strip_html(text):
    """Remove HTML tags and decode entities from a string."""
    if not text:
        return None
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    clean = clean.strip()
    return clean if clean else None


def _parse_choices(choices_str, field_type, field_name):
    """Parse the pipe-delimited select_choices_or_calculations string into a question_table list."""
    if not choices_str:
        return []
    entries = choices_str.split('|')
    question_table = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # Format: "value, description" — split on first comma only
        parts = entry.split(',', 1)
        value = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else None
        if field_type == 'checkbox':
            variable_name = f'{field_name}___{value}'
            question_table.append({'value': value, 'variable_name': variable_name, 'description': description})
        else:
            question_table.append({'value': value, 'description': description})
    return question_table


def _build_meta(data_dictionary, instruments_dict):
    """Shared logic to convert a raw data_dictionary DataFrame into the final meta DataFrame.

    Both parse_redcap_data_dict (HTML) and parse_redcap_data_dict_api (API) call this
    after building their initial data_dictionary with the same column schema.
    """
    assert not data_dictionary['name'].duplicated().any()
    meta_dict = {}
    for _, row in data_dictionary.iterrows():
        row = row.dropna().to_dict()
        meta_dict[row['name']] = row
        if row['question_type'] == 'checkbox':
            for level in row['question_table']:
                subrow = row.copy()
                subrow.pop('question_table')
                subrow['question_type'] = 'yesno'
                subrow['description'] = level['description']
                subrow['question_table'] = [{'value': '1', 'description': 'Yes'},
                                            {'value': '0', 'description': 'No'}]
                subrow['checkbox'] = True
                meta_dict[level['variable_name']] = subrow

    meta = pd.DataFrame(meta_dict).T
    meta.loc[meta['checkbox'].isna(), 'checkbox'] = False
    meta['description'] = meta['description'].fillna('No Description')

    meta.loc[(meta['question_sub_type'] == ''), 'question_sub_type'] = np.nan

    checkbox = ['checkbox']
    categorical = ['dropdown', 'radio', 'yesno']
    continuous = ['slider', 'calc']
    text = ['descriptive', 'text', 'notes']
    skipped = ['file']

    numeric_subtypes = ['number', 'integer']
    skip_subtypes = ['autocomplete', 'signature', 'email', 'phone']
    date_subtypes = ['date_dmy']

    meta['question_type'] = meta['question_type'].str.replace(', Required', '')

    meta.loc[meta['question_type'].isin(checkbox), 'question_category'] = 'checkbox'
    meta.loc[meta['question_type'].isin(text), 'question_category'] = 'text'
    meta.loc[meta['question_type'].isin(categorical), 'question_category'] = 'categorical'
    meta.loc[(meta['question_type'].isin(continuous) |
              meta['question_sub_type'].isin(numeric_subtypes)), 'question_category'] = 'numeric'
    meta.loc[meta['question_sub_type'].isin(date_subtypes), 'question_category'] = 'date'

    unhandled_mask = meta['question_category'].isna()
    meta_uh = meta.loc[unhandled_mask]
    question_types = set(meta_uh['question_type'].dropna().unique())
    handled_question_types = set(checkbox + categorical + continuous + text + skipped)
    unhandled_questions = question_types.difference(set(handled_question_types))
    if len(unhandled_questions) > 0:
        raise Exception(f'Unhandled question types: {unhandled_questions}')

    sub_question_types = set(meta_uh['question_sub_type'].dropna().unique())
    sub_handled_question_types = set(numeric_subtypes + date_subtypes + skip_subtypes)
    unhandled_sub_questions = set(sub_question_types).difference(sub_handled_question_types)
    if len(unhandled_sub_questions) > 0:
        raise Exception(f'Unhandled question subtypes: {set(sub_question_types).difference(sub_handled_question_types)}')
    meta['notes'] = pd.NA
    meta['source_variables'] = None
    return meta


def parse_redcap_data_dict_api(metadata_json, event_mapping_json=None):
    """Parse the REDCap metadata API response (JSON) into a meta DataFrame.

    Args:
        metadata_json: list of dicts from the REDCap metadata API
            (content='metadata', format='json').
        event_mapping_json: optional list of dicts from the REDCap form-event mapping API
            (content='formEventMapping', format='json'). If None, the 'event' column
            will be empty lists.

    Returns:
        pd.DataFrame: meta DataFrame with the same schema as parse_redcap_data_dict.
    """
    # Build instrument -> events mapping
    instruments_dict = {}
    if event_mapping_json:
        for entry in event_mapping_json:
            form = entry['form']
            event = entry['unique_event_name']
            if form not in instruments_dict:
                instruments_dict[form] = []
            if event not in instruments_dict[form]:
                instruments_dict[form].append(event)

    results = []
    for i, field in enumerate(metadata_json):
        name = field['field_name']
        instrument = field['form_name']
        field_type = field['field_type']

        # Required fields come through as the type itself in HTML (e.g. "radio, Required")
        # The API has a separate 'required_field' key, but we normalise to match HTML output
        question_type = field_type
        if field.get('required_field') == 'y':
            question_type = f'{field_type}, Required'

        # Parse question_sub_type from validation
        validation = field.get('text_validation_type_or_show_slider_number', '')
        question_sub_type = validation if validation else None

        # Parse min/max
        minimum = None
        maximum = None
        val_min = field.get('text_validation_min', '')
        val_max = field.get('text_validation_max', '')
        if val_min:
            try:
                minimum = float(val_min)
            except ValueError:
                pass
        if val_max:
            try:
                maximum = float(val_max)
            except ValueError:
                pass

        # Parse choices
        question_table = _parse_choices(
            field.get('select_choices_or_calculations', ''),
            field_type,
            name
        )
        # calc/slider fields use select_choices_or_calculations for formulas/labels, not choices
        if field_type in ('calc', 'slider'):
            question_table = []
        # yesno fields don't have select_choices in the API but need a standard table
        if field_type == 'yesno' and not question_table:
            question_table = [{'value': '1', 'description': 'Yes'},
                              {'value': '0', 'description': 'No'}]
        # Filter out accidental 'Parent' checkbox category
        if field_type == 'checkbox':
            question_table = [q for q in question_table if q['value'] != 'Parent']

        # Description from field_label (strip HTML)
        description = _strip_html(field.get('field_label', ''))

        # Section header
        section = field.get('section_header', '')
        section = _strip_html(section) if section else None

        # Branching logic -> restrictions
        restrictions = field.get('branching_logic', '') or None

        # Field annotation
        field_annotation = field.get('field_annotation', '') or None

        # Custom alignment
        custom_alignment = field.get('custom_alignment', '') or None

        # Question number (adjusted) from API
        adjusted_question_number = field.get('question_number', '') or None

        # Events for this instrument
        events = instruments_dict.get(instrument, [])

        results.append({
            'instrument': instrument,
            'event': events,
            'section': section,
            'question_number': i + 1,
            'name': name,
            'restrictions': restrictions,
            'description': description,
            'question_type': question_type,
            'question_sub_type': question_sub_type,
            'question_table': question_table,
            'field_annotation': field_annotation,
            'custom_alignment': custom_alignment,
            'minimum': minimum,
            'maximum': maximum,
            'adjusted_question_number': adjusted_question_number,
        })

    data_dictionary = pd.DataFrame(results)
    return _build_meta(data_dictionary, instruments_dict)


def parse_redcap_data_dict(path):
    """
    Get the redcap data dictionary from the redcap site and copy the html of the table into a file.
    This function will parse that file.

    :param path: path to html file that contains redcap data dict table.
    :return:
    """
    # Load the HTML file
    with open(path, 'r') as f:
        content = f.read()


    # Parse the HTML content using lxml's HTML parser
    html_tree = etree.HTML(content)
    instruments_table = html_tree.xpath('//table[.//span[normalize-space(text())="Instruments"]]')

    instruments_dict = {}
    for row in instruments_table[0].xpath('.//tr'):
        cols = row.xpath('.//td[@class="p-1"]')
        if len(cols) > 0:
            form_name = cols[1].xpath('text()')
            if len(form_name) > 1:
                raise Exception('Multiple forms found in instruments table.')
            instruments_dict[cols[1].xpath('text()')[0].strip()] = [i.strip() for i in cols[2].xpath('text()')]

    main_table = html_tree.xpath('//table[@id="codebook-table"]')[0]

    rows = main_table.xpath('./tbody/tr')

    table_header = rows.pop(0)
    assert table_header.attrib.get('class') == 'codebook-table-header'
    assert len(table_header.xpath('./th')) == 4

    results = []
    instrument = None
    for row in rows:
        if len(row.xpath('./th')) > 0:
            instrument = row.xpath('.//span[@style="margin-left:10px;color:#444;"]/text()')[0][1:-1]
            continue

        columns = row.xpath('./td')

        variable = columns.pop(0)
        question_number = int(variable.xpath('text()')[0])

        field_label = columns.pop(0)
        name = field_label.xpath('./code/span[@class="text-dangerrc"]/text()')[0]

        restrictions = None
        restrictions_raw = field_label.xpath('.//span[text()="Show the field ONLY if:"]/../../text()')
        if len(restrictions_raw) > 0:
            # todo parse these if needed. Probably won't be though.
            restrictions = ' '.join(restrictions_raw).strip()

        label_raw = columns.pop(0)
        section = None
        section_header_raw = label_raw.xpath('.//span[text()="Section Header: "]/../i/text()')
        if len(section_header_raw) > 0:
            section = section_header_raw[0]

        description = None
        label_txt_raw = label_raw.xpath('./text()')
        if len(label_txt_raw) > 0:
            description = ' '.join(label_txt_raw).strip()

        field_attribute = columns.pop(0)
        field_attribute_text = field_attribute.xpath('./text()')
        question_type = field_attribute_text[0].strip()

        table = field_attribute.xpath('.//table')
        question_table = []
        if len(table) > 0:
            rows = table[0].xpath('./tbody/tr')
            for row in rows:
                columns = row.xpath('./td/text()')
                if question_type != 'checkbox':
                    try:
                        second_column = columns[1]
                    except IndexError:
                        second_column = None
                    question_table.append({'value': columns[0],
                                           'description': second_column})
                else:
                    if columns[0] == 'Parent':
                        # An accidental category put into R2R.
                        continue
                    question_table.append({'value': columns[0],
                                           'variable_name': columns[1],
                                           'description': columns[2]})

        if ' (' in question_type:
            question_type_split = question_type.split()
            question_type = question_type_split[0]
            question_sub_type = question_type_split[1][1:-1]
            question_sub_type = question_sub_type.replace(')', '').strip()
        else:
            question_sub_type = None

        field_annotation = None
        custom_alignment = None
        maximum = None
        minimum = None
        adjusted_question_number = None
        span = field_attribute.xpath('./span')
        for s in span:
            if s.xpath('./text()')[0] == 'Field Annotation':
                annotation_raw = s.xpath('../span[text()="Field Annotation"]/following-sibling::text()[1]')
                field_annotation = annotation_raw[0][2:]
            elif s.xpath('./text()')[0] == 'Custom alignment:':
                alignment_raw = s.xpath('../span[text()="Custom alignment:"]/following-sibling::text()[1]')
                custom_alignment = alignment_raw[0][1:]
            elif s.xpath('./text()')[0] == 'Max:':
                max_raw = s.xpath('../span[text()="Max:"]/following-sibling::text()[1]')
                maximum = float(max_raw[0][1:-1])
            elif s.xpath('./text()')[0] == 'Min:':
                min_raw = s.xpath('../span[text()="Min:"]/following-sibling::text()[1]')
                minimum = float(re.findall(r'\d+', min_raw[0])[0])
            elif 'Question number:' in s.xpath('./text()')[0]:
                adjusted_question_number = s.xpath('following-sibling::text()[1]')[0].replace(' ', '')

        results.append({'instrument': instrument,
                        'event': instruments_dict[instrument],
                        'section': section,
                        'question_number': question_number,
                        'name': name,
                        'restrictions': restrictions,
                        'description': description,
                        'question_type': question_type,
                        'question_sub_type': question_sub_type,
                        'field_annotation': field_annotation,
                        'question_table': question_table,
                        'custom_alignment': custom_alignment,
                        'minimum': minimum,
                        'maximum': maximum,
                        'adjusted_question_number': adjusted_question_number})

    data_dictionary = pd.DataFrame(results)
    return _build_meta(data_dictionary, instruments_dict)
